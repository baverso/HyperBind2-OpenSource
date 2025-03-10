"""Multi-Epoch ESM3 LoRA Validation Script."""
import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import time
import math
import gc
from datetime import datetime
from Bio.PDB import PDBParser
from IPython.display import display, HTML
from esm.sdk.api import ESMProtein, ProteinComplex, GenerationConfig

sys.path.append("scripts")

# Configure basic logging to file
log_file = f"multi_epoch_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger("validation")


def log_print(message):
    """Print to notebook and also log to file."""
    print(message)
    logger.info(message)


log_print(f"Multi-Epoch ESM3 LoRA Validation Script - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Logging to file: {log_file}")


class LoRALayer(torch.nn.Module):
    """Custom LoRA implementation for ESM3 model layers."""

    def __init__(
        self,
        base_layer: torch.nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ):
        """Initialize LoRA layer with low-rank matrices.

        Args:
            base_layer: The original linear layer to wrap
            rank: Rank of the LoRA update
            alpha: Scaling factor
            dropout: Dropout rate applied between LoRA A and B
        """
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Collect dimensions from the original layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Create LoRA low-rank matrices A and B
        self.lora_A = torch.nn.Parameter(torch.zeros(self.in_features, self.rank))
        self.lora_B = torch.nn.Parameter(torch.zeros(self.rank, self.out_features))

        # Dropout for LoRA branch
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        self.reset_parameters()

        # Freeze the *base_layer* so it does not train
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_parameters(self) -> None:
        """Initialize LoRA matrices (A and B)."""
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine base layer output with scaled LoRA branch output."""
        # Original linear pass
        base_output = self.base_layer(x)

        # LoRA path: x -> A -> dropout -> B -> scale
        lora_output = self.dropout(x @ self.lora_A) @ self.lora_B

        # Combine outputs
        return base_output + (lora_output * self.scaling)


def apply_lora_to_model(
    model: torch.nn.Module,
    target_modules: list,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
):
    """Apply LoRA to specified nn.Linear layers by name.

    Args:
        model: The model to modify in-place
        target_modules: List of module names to apply LoRA to
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout for LoRA branch

    Returns:
        Modified model and list of modified modules
    """
    log_print(f"Applying LoRA with rank={rank}, alpha={alpha}, dropout={dropout}")
    modified_modules = []
    parent_modules = {}
    
    # Find the target modules
    for name, module in model.named_modules():
        if name in target_modules and isinstance(module, torch.nn.Linear):
            log_print(f"Found target module: {name}")
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            
            # Traverse to find the parent module
            parent_module = model
            if parent_name:
                for part in parent_name.split("."):
                    parent_module = getattr(parent_module, part)
                    
            if parent_name not in parent_modules:
                parent_modules[parent_name] = []
            parent_modules[parent_name].append((attr_name, module))
            modified_modules.append(name)
            
    # Replace each target Linear with LoRALayer
    for parent_name, attr_list in parent_modules.items():
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)
                
        for attr_name, module in attr_list:
            try:
                lora_layer = LoRALayer(
                    base_layer=module, rank=rank, alpha=alpha, dropout=dropout
                )
                setattr(parent, attr_name, lora_layer)
                log_print(f"✅ Successfully replaced {parent_name}.{attr_name} with LoRA layer")
            except Exception as e:
                log_print(f"❌ Error creating LoRA layer for {parent_name}.{attr_name}: {str(e)}")
                
    return model, modified_modules


def create_esm3_model():
    """Return a fresh pretrained ESM3 model instance."""
    log_print("Creating ESM3 model...")
    try:
        from esm.pretrained import ESM3_sm_open_v0
        model = ESM3_sm_open_v0()
        log_print("✅ ESM3 model created")
        return model
    except Exception as e:
        log_print(f"❌ Error creating ESM3 model: {str(e)}")
        raise


def load_base_model():
    """Load the base ESM3 model without LoRA."""
    log_print("\n=== Loading Base ESM3 Model (No LoRA) ===")
    
    # Create base model
    model = create_esm3_model()
    
    # Load weights
    model_path = "/home/jupyter/DATA/model_weights/esm3_complete/esm3_sm_open_v1_state_dict.pt"
    log_print(f"Loading weights from: {model_path}")
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Set to eval mode
    model.eval()
    log_print("Model set to evaluation mode")
    
    return model


def load_model_with_lora(weights_path, lora_rank=16, lora_alpha=32, lora_dropout=0.1):
    """Load a model with LoRA and specific weights.
    
    Args:
        weights_path: Path to checkpoint file
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout parameter
        
    Returns:
        Model with LoRA applied and weights loaded
    """
    log_print(f"\n=== Loading Model with LoRA weights: {os.path.basename(weights_path)} ===")
    
    # Create base model
    model = create_esm3_model()
    
    # Load original ESM3 weights
    model_path = "/home/jupyter/DATA/model_weights/esm3_complete/esm3_sm_open_v1_state_dict.pt"
    log_print(f"Loading base ESM3 weights from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Apply LoRA
    target_modules = [
        # Block 0 MHA
        "transformer.blocks.0.attn.layernorm_qkv.1",
        "transformer.blocks.0.attn.out_proj",
        # Block 1 MHA
        "transformer.blocks.1.attn.layernorm_qkv.1",
        "transformer.blocks.1.attn.out_proj",
        # Block 0 geometric attention
        "transformer.blocks.0.geom_attn.proj",
        "transformer.blocks.0.geom_attn.out_proj",
        # Block 1 geometric attention
        "transformer.blocks.1.geom_attn.proj",
        "transformer.blocks.1.geom_attn.out_proj",
    ]
    
    # Apply LoRA
    model, modified_modules = apply_lora_to_model(
        model,
        target_modules=target_modules,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout
    )
    
    # Load LoRA weights
    log_print(f"Loading LoRA weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    
    if "model_state_dict" in checkpoint:
        log_print("Loading from model_state_dict key")
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        log_print("Loading directly from checkpoint")
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    
    log_print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    # Set to eval mode
    model.eval()
    log_print("Model set to evaluation mode")
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return model


def prepare_test_data():
    """Prepare test data once for all models.
    
    Returns:
        Tuple of (protein_list, ground_truth_structures, pdb_ids)
    """
    log_print("\n=== Preparing Test Data ===")
    
    # Get test files
    test_directory = "/home/jupyter/DATA/hyperbind_train/sabdab/all_structures/train-test-split/"
    log_print(f"Looking for test files in: {test_directory}")
    test_pdb_files = [f for f in os.listdir(test_directory) if f.endswith("_test.pdb")]
    log_print(f"Found {len(test_pdb_files)} test PDB files")
    
    # Import necessary modules
    try:
        sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "scripts")))
        from esm.sdk.api import ESMProtein, GenerationConfig
        from pdb2esm import detect_and_process_structure
        log_print("Successfully imported ESM modules")
    except ImportError as e:
        log_print(f"❌ Could not import ESM modules: {e}")
        raise
    
    # Process test dataset
    log_print("Processing test PDB files into ESMProtein objects...")
    protein_list = []
    parser = PDBParser(QUIET=True)
    ground_truth_structures = []
    pdb_ids = []
    
    for i, filename in enumerate(test_pdb_files):
        pdb_path = os.path.join(test_directory, filename)
        pdb_id = filename.replace("_test.pdb", "")
        pdb_ids.append(pdb_id)
        log_print(f"Processing [{i+1}/{len(test_pdb_files)}] {pdb_id}")
        
        # Load as ESMProtein
        try:
            protein = detect_and_process_structure(pdb_path)
            if protein is None:
                log_print(f"❌ Failed to process {pdb_id} as ESMProtein")
                continue
            protein_list.append(protein)
            
            # Also load as BioPython structure for RMSD calculation
            structure = parser.get_structure(pdb_id, pdb_path)
            ground_truth_structures.append(structure)
            log_print(f"✅ Processed {pdb_id}")
        except Exception as e:
            log_print(f"❌ Error processing {pdb_id}: {str(e)}")
    
    log_print(f"Successfully processed {len(protein_list)}/{len(test_pdb_files)} proteins")
    
    return protein_list, ground_truth_structures, pdb_ids


def evaluate_model_rmsd(model, protein_list, ground_truth_structures, pdb_ids, model_name):
    """Evaluate a model and return RMSD results.
    
    Args:
        model: ESM3 model (with or without LoRA)
        protein_list: List of ESMProtein objects
        ground_truth_structures: List of BioPython structures
        pdb_ids: List of PDB IDs
        model_name: Name to identify this model in results
        
    Returns:
        DataFrame with RMSD results or None if error
    """
    log_print(f"\n=== Evaluating Model: {model_name} ===")
    
    # Import necessary modules
    try:
        from esm.sdk.api import GenerationConfig
        from rmsd import compute_alignment_rmsd_table
        log_print("Successfully imported evaluation modules")
    except ImportError as e:
        log_print(f"❌ Could not import evaluation modules: {e}")
        return None
    
    # Set up directories - use model-specific subfolder
    base_folder = "/home/jupyter/DATA/hyperbind_inferred/inferred_structures"
    inferred_folder = os.path.join(base_folder, model_name)
    os.makedirs(inferred_folder, exist_ok=True)
    
    # Prepare generation configuration
    config = GenerationConfig(track="structure", schedule="cosine")
    configs = [config] * len(protein_list)
    
    # Copy proteins to avoid modifying the originals
    copied_proteins = []
    for protein in protein_list:
        # Create a copy by re-processing
        copied = ESMProtein(sequence=protein.sequence)
        copied_proteins.append(copied)
    
    # Generate structures
    log_print(f"Generating structures for {len(copied_proteins)} proteins...")
    start_time = time.time()
    
    with torch.no_grad():
        try:
            output = model.batch_generate(inputs=copied_proteins, configs=configs)
            log_print(f"✅ Successfully generated {len(output)} structures")
        except Exception as e:
            log_print(f"❌ Error in batch generation: {e}")
            log_print("Falling back to individual generation...")
            
            # Try generating one by one
            output = []
            for i, (protein, config) in enumerate(zip(copied_proteins, configs)):
                try:
                    log_print(f"Generating structure [{i+1}/{len(copied_proteins)}]")
                    result = model.generate(protein, config)
                    output.append(result)
                    log_print(f"✅ Generated structure {i+1}")
                except Exception as e:
                    log_print(f"❌ Failed to generate structure {i+1}: {e}")
                    output.append(None)
    
    generation_time = time.time() - start_time
    log_print(f"Structure generation completed in {generation_time:.1f} seconds")
    
    # Save generated structures
    log_print("Saving generated structures...")
    saved_structures = []
    
    for i, (protein, pdb_id) in enumerate(zip(output, pdb_ids[:len(output)])):
        if protein is None:
            log_print(f"❌ No structure for {pdb_id}")
            continue
            
        try:
            pdb_string = protein.to_pdb_string()
            file_path = os.path.join(inferred_folder, f"{pdb_id}_inferred.pdb")
            with open(file_path, "w") as f:
                f.write(pdb_string)
            saved_structures.append((pdb_id, file_path))
            log_print(f"✅ Saved structure for {pdb_id}")
        except Exception as e:
            log_print(f"❌ Error saving structure for {pdb_id}: {e}")
    
    # Load saved structures for RMSD calculation
    log_print("Loading inferred structures for RMSD calculation...")
    parser = PDBParser(QUIET=True)
    inferred_structures = []
    structure_pdb_ids = []
    
    for pdb_id, file_path in saved_structures:
        try:
            structure = parser.get_structure(pdb_id, file_path)
            inferred_structures.append(structure)
            structure_pdb_ids.append(pdb_id)
            log_print(f"✅ Loaded inferred structure for {pdb_id}")
        except Exception as e:
            log_print(f"❌ Error loading inferred structure for {pdb_id}: {e}")
    
    # Get matching ground truth structures
    matching_ground_truth = []
    matching_ids = []
    
    for pdb_id in structure_pdb_ids:
        matched = False
        for i, orig_id in enumerate(pdb_ids):
            if pdb_id == orig_id:
                matching_ground_truth.append(ground_truth_structures[i])
                matching_ids.append(pdb_id)
                matched = True
                break
        if not matched:
            log_print(f"❌ Could not find matching ground truth for {pdb_id}")
    
    # Compute RMSD
    if len(matching_ground_truth) > 0 and len(inferred_structures) > 0:
        log_print(f"Computing RMSD for {len(inferred_structures)} structures...")
        try:
            results = compute_alignment_rmsd_table(matching_ground_truth, inferred_structures)
            results_df = pd.DataFrame(results)
            
            # Add PDB IDs and model name
            if 'Structure' not in results_df.columns:
                results_df['Structure'] = matching_ids
            
            results_df['Model'] = model_name
            
            log_print("✅ RMSD calculation complete")
            
            # Calculate summary statistics
            mae = results_df["Global RMSD"].mean()
            log_print(f"Mean Global RMSD: {mae:.4f}")
            
            return results_df
        except Exception as e:
            log_print(f"❌ Error calculating RMSD: {e}")
            return None
    else:
        log_print("❌ No structures available for RMSD calculation")
        return None


def save_results_csv(all_results, filename="rmsd_comparison.csv"):
    """Save all results to a CSV file.
    
    Args:
        all_results: DataFrame with results
        filename: Output filename
        
    Returns:
        Filename if successful, None otherwise
    """
    if all_results is not None:
        all_results.to_csv(filename, index=False)
        log_print(f"Results saved to {filename}")
        return filename
    return None


def run_all_validations():
    """Run validation on base model and all epochs.
    
    Returns:
        Tuple of (combined_results, summary) or (None, None) if error
    """
    # Clear GPU memory if available
    if torch.cuda.is_available():
        log_print("Clearing GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        log_print(f"GPU: {torch.cuda.get_device_name(0)}")
        log_print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    # Prepare data once
    log_print("Preparing test data...")
    protein_list, ground_truth_structures, pdb_ids = prepare_test_data()
    
    # Define all models to evaluate
    models_to_evaluate = [
        {"name": "base", "path": None},  # Base model without LoRA
        {"name": "epoch1", "path": "/home/jupyter/DATA/esm_finetuning/new_lora_checkpoints_03_10/checkpoint_epoch_1.pt"},
        {"name": "epoch2", "path": "/home/jupyter/DATA/esm_finetuning/new_lora_checkpoints_03_10/checkpoint_epoch_2.pt"},
        {"name": "epoch3", "path": "/home/jupyter/DATA/esm_finetuning/new_lora_checkpoints_03_10/checkpoint_epoch_3.pt"},
        {"name": "epoch4", "path": "/home/jupyter/DATA/esm_finetuning/new_lora_checkpoints_03_10/checkpoint_epoch_4.pt"},
        {"name": "epoch5", "path": "/home/jupyter/DATA/esm_finetuning/new_lora_checkpoints_03_10/checkpoint_epoch_5.pt"},
    ]
    
    # Store all results
    all_results = []
    
    # Evaluate each model
    for model_info in models_to_evaluate:
        model_name = model_info["name"]
        weights_path = model_info["path"]
        
        try:
            # Load appropriate model
            if model_name == "base":
                model = load_base_model()
            else:
                model = load_model_with_lora(weights_path)
            
            # Evaluate model
            results_df = evaluate_model_rmsd(
                model=model,
                protein_list=protein_list,
                ground_truth_structures=ground_truth_structures,
                pdb_ids=pdb_ids,
                model_name=model_name
            )
            
            if results_df is not None:
                all_results.append(results_df)
                
                # Save individual results too
                results_df.to_csv(f"rmsd_{model_name}.csv", index=False)
                log_print(f"Saved results for {model_name} to rmsd_{model_name}.csv")
            
            # Clear memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            log_print(f"❌ Error evaluating {model_name}: {e}")
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        filename = save_results_csv(combined_results)
        
        # Create summary table
        summary = combined_results.groupby('Model')['Global RMSD'].agg(['mean', 'std', 'min', 'max']).reset_index()
        summary.columns = ['Model', 'Mean RMSD', 'Std Dev', 'Min RMSD', 'Max RMSD']
        
        # Save summary
        summary.to_csv("rmsd_summary.csv", index=False)
        log_print("Summary saved to rmsd_summary.csv")
        
        # Display summary
        log_print("\n=== RMSD Summary ===")
        display(summary)
        
        return combined_results, summary
    else:
        log_print("❌ No results collected")
        return None, None


# --- Main execution ---
if __name__ == "__main__":
    all_results, summary = run_all_validations()