#!/usr/bin/env python
"""
esm3_validation.py

This script loads a pretrained ESM3 model (unless an existing model is provided),
processes a test set of PDB files into ESMProtein objects, generates predicted structures
(using a GenerationConfig that forces structure generation solely from sequence),
saves the inferred structures, and evaluates the predictions by comparing them against the
ground‐truth structures via RMSD calculations.

Usage (from the command line):
    python esm3_validation.py --cli --weights_path /path/to/weights.pt

You can also import the main functions into a Jupyter notebook and pass in your own model.
"""

import os
import sys
import torch
import pandas as pd
from Bio.PDB import PDBParser

# Ensure the scripts directory (which contains pdb2esm.py) is in the Python path.
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "scripts")))
from pdb2esm import read_monomer_structure, read_multimer_structure, detect_and_process_structure
from esm.sdk.api import ESMProtein, ProteinComplex, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain
from rmsd import compute_alignment_rmsd_table  # Ensure you have an RMSD module available


def create_esm3_model():
    """Returns a fresh pretrained ESM3 model instance using the helper from esm.pretrained."""
    from esm.pretrained import ESM3_sm_open_v0
    model = ESM3_sm_open_v0()  # Returns the model object.
    return model


def run_validation(weights_path="/home/jupyter/DATA/model_weights/esm3_complete/esm3_sm_open_v1_state_dict.pt",
                   esm_model=None):
    """
    Runs the full test validation.
    
    If esm_model is provided (an already loaded ESM model), it will be used directly;
    otherwise, a fresh model instance will be created and weights loaded from weights_path.
    
    Returns:
        results_df: DataFrame with RMSD results.
        mae: Mean Absolute Error computed on Global RMSD.
    """
    # Directories: adjust these paths as needed.
    test_directory = "/home/jupyter/DATA/hyperbind_train/sabdab/all_structures/train-test-split/"
    output_directory = "/home/jupyter/DATA/hyperbind_train/sabdab/all_structures/esm3_predictions/"
    os.makedirs(output_directory, exist_ok=True)
    
    inferred_folder = "/home/jupyter/DATA/hyperbind_inferred/inferred_structures"
    os.makedirs(inferred_folder, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use provided model if available; otherwise, instantiate and load weights.
    if esm_model is None:
        model = create_esm3_model()
        print(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = esm_model
    model.to(device)
    model.eval()
    
    # ---------------------
    # Process Test Dataset: load ESMProtein objects from PDB files.
    # ---------------------
    test_pdb_files = [f for f in os.listdir(test_directory) if f.endswith("_test.pdb")]
    protein_list = []
    for filename in test_pdb_files:
        pdb_path = os.path.join(test_directory, filename)
        pdb_id = filename.replace("_test.pdb", "")
        protein = detect_and_process_structure(pdb_path)
        if protein is None:
            print(f"Warning: Failed to process {pdb_path}")
            continue
        protein_list.append(protein)
    
    # ---------------------
    # Parse the ground-truth structures using Biopython for RMSD evaluation.
    # ---------------------
    ground_truth_structures = []
    parser = PDBParser(QUIET=True)
    for filename in test_pdb_files:
        pdb_path = os.path.join(test_directory, filename)
        pdb_id = filename.replace("_test.pdb", "")
        try:
            structure = parser.get_structure(pdb_id, pdb_path)
            ground_truth_structures.append(structure)
            print(f"✅ Parsed ground-truth {pdb_id} successfully.")
        except Exception as e:
            print(f"❌ Error parsing ground-truth {pdb_id}: {e}")
    
    # ---------------------
    # Prepare generation configuration.
    # ---------------------
    # Force generation to rely solely on the sequence by clearing out structure fields.
    config = GenerationConfig(track="structure", schedule="cosine")
    configs = [config] * len(protein_list)
    for protein in protein_list:
        protein.coordinates = None
        protein.sasa = None
        protein.function_annotations = None
    
    # ---------------------
    # Run structure generation.
    # ---------------------
    with torch.no_grad():
        output = model.batch_generate(inputs=protein_list, configs=configs)
    
    # Create a list of pdb IDs for naming inferred files.
    pdb_handles = [filename.replace("_test.pdb", "") for filename in test_pdb_files]
    
    # ---------------------
    # Save inferred structures.
    # ---------------------
    for name, protein in zip(pdb_handles, output):
        pdb_string = protein.to_pdb_string()  # Assumes method returns a valid PDB string.
        file_path = os.path.join(inferred_folder, f"{name}_inferred.pdb")
        with open(file_path, "w") as f:
            f.write(pdb_string)
        print(f"Saved inferred structure to {file_path}")
    
    # ---------------------
    # Parse the inferred structures using Biopython.
    # ---------------------
    inferred_pdb_files = [f for f in os.listdir(inferred_folder) if f.endswith("_inferred.pdb")]
    inferred_list = []
    for filename in inferred_pdb_files:
        pdb_path = os.path.join(inferred_folder, filename)
        pdb_id = filename.replace("_inferred.pdb", "")
        try:
            structure = parser.get_structure(pdb_id, pdb_path)
            inferred_list.append(structure)
            print(f"✅ Parsed inferred {pdb_id} successfully.")
        except Exception as e:
            print(f"❌ Error parsing inferred {pdb_id}: {e}")
    
    # ---------------------
    # Compute RMSD and report results.
    # ---------------------
    results = compute_alignment_rmsd_table(ground_truth_structures, inferred_list)
    results_df = pd.DataFrame(results)
    
    mae = (results_df["Global RMSD"] - results_df["Global RMSD"].mean()).abs().mean()
    print("MAE on Global RMSD column:", mae)
    
    return results_df, mae


def main():
    """
    Main function to run the complete test validation from the CLI.
    Also suitable for import into a Jupyter notebook.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Run complete test validation for ESM3 structure prediction."
    )
    parser.add_argument("--cli", action="store_true", help="Run as CLI command")
    parser.add_argument("--weights_path", type=str,
                        default="/home/jupyter/DATA/model_weights/esm3_complete/esm3_sm_open_v1_state_dict.pt",
                        help="Path to the model weights file.")
    args = parser.parse_args()
    
    # When running via CLI, we always create a fresh model instance.
    results_df, mae = run_validation(weights_path=args.weights_path, esm_model=None)
    print("Validation complete.")
    print(results_df)
    print("MAE on RMSD:", mae)


if __name__ == "__main__":
    main()