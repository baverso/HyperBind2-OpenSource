#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import gc
import time
import math
import random
import numpy as np

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt

from Bio import PDB
from Bio.PDB import PDBParser

import esm
from esm.pretrained import ESM3_sm_open_v0, ESM3_structure_encoder_v0
from esm.tokenization import get_esm3_model_tokenizers
from esm.utils.encoding import tokenize_sequence, tokenize_structure
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Force synchronous CUDA execution (sometimes helpful for debugging)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Clear CPU/GPU caches
gc.collect()
torch.cuda.empty_cache()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get ESM-3 tokenizers
tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
sequence_tokenizer = tokenizers.sequence
structure_tokenizer = tokenizers.structure

# Initialize structure encoder on CPU
structure_encoder = ESM3_structure_encoder_v0(device="cpu")


class StructureData:
    """
    Simple container for amino acid sequence and 3D coordinates.
    """

    def __init__(
        self,
        sequence: str,
        atom_coords: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            sequence (str): Amino acid sequence in single-letter codes.
            atom_coords (torch.Tensor): [n_res, n_atoms, 3] 3D coordinates.
            atom_mask (torch.Tensor): [n_res, n_atoms] boolean mask for valid atoms.
        """
        self.sequence = sequence
        self.atom_coords = atom_coords
        self.atom_mask = (
            atom_mask
            if atom_mask is not None
            else torch.ones_like(atom_coords[..., 0], dtype=torch.bool)
        )


def process_pdb_file(file_path: str) -> Optional["StructureData"]:
    """
    Process a PDB file and return sequence + coordinates for N, CA, C, O, CB.
    Ignores residues not in standard amino acids (3-letter code -> 1-letter).
    Handles glycine CB as missing if not present.

    Args:
        file_path (str): Full path to .pdb file.

    Returns:
        StructureData or None if no valid residues were found.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)

    # 3-letter to 1-letter conversion dictionary
    three_to_one = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
    }

    # Atoms to extract in order
    atoms_to_extract = ["N", "CA", "C", "O", "CB"]

    valid_residues = []
    sequence_chars = []

    # Collect valid residues
    for model in structure:
        for chain in model:
            for residue in chain:
                if ("CA" in residue) and (residue.get_resname() in three_to_one):
                    valid_residues.append(residue)
                    sequence_chars.append(three_to_one[residue.get_resname()])

    if not valid_residues:
        return None

    sequence_str = "".join(sequence_chars)
    n_res = len(sequence_str)
    n_atoms = len(atoms_to_extract)

    # Initialize coordinate arrays
    coords = np.zeros((n_res, n_atoms, 3), dtype=np.float32)
    atom_mask = np.zeros((n_res, n_atoms), dtype=bool)

    # Extract coords and build mask
    for i, residue in enumerate(valid_residues):
        for j, atom_name in enumerate(atoms_to_extract):
            if atom_name in residue:
                coords[i, j] = residue[atom_name].get_coord()
                atom_mask[i, j] = True
            elif atom_name == "CB" and residue.get_resname() == "GLY":
                # Glycine doesn't have CB; just skip
                continue

    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    mask_tensor = torch.tensor(atom_mask, dtype=torch.bool)

    return StructureData(
        sequence=sequence_str, atom_coords=coords_tensor, atom_mask=mask_tensor
    )


class LoRALayer(nn.Module):
    """
    Custom LoRA implementation for ESM3 model layers.
    LoRA adapts a base Linear layer with low-rank matrices.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ):
        """
        Args:
            base_layer (nn.Linear): The original linear layer to wrap.
            rank (int): Rank of the LoRA update.
            alpha (int): Scaling factor (alpha = r * multiplier).
            dropout (float): Dropout rate applied between LoRA A and B.
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
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features))

        # Dropout for LoRA branch
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

        # Freeze the *base_layer* so it does not train
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_parameters(self) -> None:
        """
        Initialize LoRA matrices (A and B).
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combines the base layer output + scaled LoRA branch output.
        """
        # Original linear pass
        base_output = self.base_layer(x)

        # LoRA path: x -> A -> dropout -> B -> scale
        lora_output = self.dropout(x @ self.lora_A) @ self.lora_B

        # Combine outputs
        return base_output + (lora_output * self.scaling)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
) -> Tuple[nn.Module, List[str]]:
    """
    Apply LoRA to specified nn.Linear layers by name.

    Args:
        model (nn.Module): The model to modify in-place.
        target_modules (List[str]): List of fully-qualified module names.
        rank (int): LoRA rank.
        alpha (int): LoRA alpha scaling factor.
        dropout (float): Dropout for LoRA branch.

    Returns:
        (model, modified_modules): The updated model and a list of modules replaced.
    """
    modified_modules = []
    parent_modules = {}

    # Find the target modules
    for name, module in model.named_modules():
        if name in target_modules and isinstance(module, nn.Linear):
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
            lora_layer = LoRALayer(
                base_layer=module, rank=rank, alpha=alpha, dropout=dropout
            )
            setattr(parent, attr_name, lora_layer)

    return model, modified_modules


def save_lora_weights(model: nn.Module, path: str) -> None:
    """
    Save only the LoRA parameters to disk.

    Args:
        model (nn.Module): Model with LoRA modules attached.
        path (str): Path to the file.
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        # Only LoRA params have requires_grad=True after injection
        if param.requires_grad:
            lora_state_dict[name] = param.data.cpu()

    torch.save(lora_state_dict, path)
    print(f"LoRA weights saved to {path} ({len(lora_state_dict)} parameters)")


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """
    Load LoRA parameters from a file into an existing model.

    Args:
        model (nn.Module): Model to which LoRA parameters should be loaded.
        path (str): Path to the saved LoRA weights file.

    Returns:
        nn.Module: Model with LoRA weights loaded.
    """
    lora_state_dict = torch.load(path)
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
    print(f"Loaded LoRA weights from {path}")
    print(f"  Missing parameters: {len(missing)}")
    print(f"  Unexpected parameters: {len(unexpected)}")
    return model


class AntibodyDataset(Dataset):
    """
    A dataset that loads PDB files of antibody structures and
    returns tokenized sequences + structure tokens.
    """

    def __init__(self, pdb_dir: str):
        """
        Args:
            pdb_dir (str): Directory containing .pdb files.
        """
        self.pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        print(f"Found {len(self.pdb_files)} PDB files")

    def __len__(self) -> int:
        return len(self.pdb_files)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Union[str, torch.Tensor]]]:
        """
        Retrieves one item of data: sequence tokens, structure tokens, and filepath.

        Args:
            idx (int): Index of the PDB file.

        Returns:
            Dict with "sequence", "structure", "file", or None if there's an error.
        """
        pdb_file = self.pdb_files[idx]
        try:
            structure_data = process_pdb_file(pdb_file)
            if structure_data is None:
                print(f"Failed to process {pdb_file}")
                return None

            # Tokenize sequence
            try:
                input_seq = tokenize_sequence(
                    structure_data.sequence, sequence_tokenizer, add_special_tokens=True
                )
            except Exception as exc:
                print(f"Sequence tokenization failed for {pdb_file}: {str(exc)}")
                return None

            # Tokenize structure
            try:
                _, _, structure_tokens = tokenize_structure(
                    structure_data.atom_coords,
                    structure_encoder,
                    structure_tokenizer,
                    reference_sequence=structure_data.sequence,
                    add_special_tokens=True,
                )

                # Ensure valid range [0..4095], -100 for padding
                valid_tokens = structure_tokens != -100
                structure_tokens[valid_tokens] = torch.clamp(
                    structure_tokens[valid_tokens], 0, 4095
                )

            except Exception as exc:
                print(f"Structure tokenization failed for {pdb_file}: {str(exc)}")
                return None

            return {
                "sequence": input_seq,
                "structure": structure_tokens,
                "file": pdb_file,
            }

        except Exception as exc:
            print(f"Error processing {pdb_file}: {str(exc)}")
            return None


def custom_collate_fn(
    batch: List[Optional[Dict[str, torch.Tensor]]]
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Custom collate function to handle variable-length sequence/structure tokens
    and skip invalid items.
    """
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if not batch:
        print("No valid items in batch")
        return None

    try:
        sequences = [item["sequence"] for item in batch]
        structures = [item["structure"] for item in batch]
        files = [item["file"] for item in batch]

        # Pad sequences and structures
        padded_sequences = nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0
        )
        padded_structures = nn.utils.rnn.pad_sequence(
            structures, batch_first=True, padding_value=-100
        )

        return {
            "sequence": padded_sequences,
            "structure": padded_structures,
            "files": files,
        }
    except Exception as exc:
        print(f"Error in collate_fn: {str(exc)}")
        return None


def print_gpu_memory() -> None:
    """
    Print current GPU memory usage if CUDA is available.
    """
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory allocated: {allocated_gb:.2f} GB")
        print(f"GPU Memory reserved: {reserved_gb:.2f} GB")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[ReduceLROnPlateau],
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_losses: List[float],
    val_losses: List[float],
    filename: str,
) -> None:
    """
    Save model checkpoint and training state.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        filename,
    )
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[ReduceLROnPlateau],
    filename: str,
) -> Tuple[int, List[float], List[float]]:
    """
    Load model checkpoint and training state.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        sched_state = checkpoint["scheduler_state_dict"]
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], train_losses, val_losses


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    """
    Evaluate model on a validation set.
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            try:
                input_seq = batch["sequence"].to(device)
                target_structure = batch["structure"].to(device)

                output = model(
                    sequence_tokens=input_seq, structure_tokens=target_structure
                )
                pred_logits = output.structure_logits

                n_classes = pred_logits.size(-1)
                if target_structure.max() >= n_classes:
                    # Some targets are out of range
                    print("Warning: target values exceed number of classes")
                    continue

                loss = loss_fn(
                    pred_logits.view(-1, n_classes), target_structure.view(-1)
                )
                total_loss += loss.item()
                valid_batches += 1

            except Exception as exc:
                print(f"Error in validation batch {batch_idx}: {str(exc)}")
                continue

    if valid_batches > 0:
        return total_loss / valid_batches
    return float("inf")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[ReduceLROnPlateau],
    loss_fn: nn.Module,
    num_epochs: int,
    device: torch.device,
    checkpoint_dir: str,
    start_epoch: int = 0,
    train_losses: Optional[List[float]] = None,
    val_losses: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Train model with optional validation and checkpointing.
    Saves a checkpoint at the end of every epoch.
    """
    model.train()
    print("Starting training...")

    if train_losses is None:
        train_losses = []
    if val_losses is None:
        val_losses = []

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Early stopping parameters
    patience = 10
    best_val_loss = float("inf")
    no_improve_count = 0

    scaler = GradScaler()

    start_time = time.time()
    batch_times = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        valid_batches = 0

        # ----- Training Loop -----
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            try:
                batch_start_time = time.time()

                input_seq = batch["sequence"].to(device)
                target_structure = batch["structure"].to(device)

                optimizer.zero_grad()

                # Mixed-precision forward
                with autocast():
                    output = model(
                        sequence_tokens=input_seq, structure_tokens=target_structure
                    )
                    pred_logits = output.structure_logits

                    n_classes = pred_logits.size(-1)
                    if target_structure.max() >= n_classes:
                        print("Warning: target values exceed number of classes")
                        print(
                            f"Max target: {target_structure.max()}, "
                            f"Num classes: {n_classes}"
                        )
                        continue

                    loss = loss_fn(
                        pred_logits.view(-1, n_classes), target_structure.view(-1)
                    )

                # Backward + gradient scale
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                valid_batches += 1

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)

                # ETA calculation
                if len(batch_times) > 10:
                    avg_batch_time = sum(batch_times[-10:]) / 10
                    remaining_batches = len(train_loader) - (batch_idx + 1)
                    remaining_time = avg_batch_time * remaining_batches
                    eta_ts = time.time() + remaining_time
                    eta = datetime.fromtimestamp(eta_ts).strftime("%H:%M:%S")
                else:
                    eta = "calculating..."

                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(
                        f"Epoch {epoch + 1}/{start_epoch + num_epochs}, "
                        f"Batch {batch_idx + 1}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}, ETA: {eta}"
                    )
                    print_gpu_memory()

            except Exception as exc:
                print(f"Error in training batch {batch_idx}: {str(exc)}")
                continue

        epoch_duration = time.time() - epoch_start_time

        if valid_batches > 0:
            avg_train_loss = total_loss / valid_batches
            train_losses.append(avg_train_loss)

            print("\nValidating model...")
            val_loss = evaluate_model(model, val_loader, loss_fn, device)
            val_losses.append(val_loss)

            if scheduler:
                scheduler.step(val_loss)

            print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Plot and save learning curves
            plot_learning_curves(
                train_losses,
                val_losses,
                os.path.join(checkpoint_dir, f"learning_curves_epoch_{epoch + 1}.png"),
            )

            # SAVE A CHECKPOINT EVERY EPOCH
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                avg_train_loss,
                val_loss,
                train_losses,
                val_losses,
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

            # Check improvement -> save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    avg_train_loss,
                    val_loss,
                    train_losses,
                    val_losses,
                    os.path.join(checkpoint_dir, "best_model.pt"),
                )
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} epochs")

            # Early stopping
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        else:
            # No valid batches in this epoch
            print(f"Epoch {epoch + 1} completed with no valid batches")

    total_training_time = time.time() - start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    return train_losses, val_losses


def plot_learning_curves(
    train_losses: List[float], val_losses: List[float], save_path: Optional[str] = None
) -> None:
    """
    Plot and optionally save training/validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Annotate minimums
    if train_losses:
        min_train_loss = min(train_losses)
        min_train_epoch = train_losses.index(min_train_loss) + 1
        plt.text(
            0.02,
            0.95,
            f"Min Train Loss: {min_train_loss:.4f} (Epoch {min_train_epoch})",
            transform=plt.gca().transAxes,
            fontsize=9,
        )
    if val_losses:
        min_val_loss = min(val_losses)
        min_val_epoch = val_losses.index(min_val_loss) + 1
        plt.text(
            0.02,
            0.90,
            f"Min Val Loss: {min_val_loss:.4f} (Epoch {min_val_epoch})",
            transform=plt.gca().transAxes,
            fontsize=9,
        )

    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to {save_path}")
    plt.close()


def main() -> None:
    """
    Main function to prepare data, create model, apply LoRA, and start training.
    """
    # Set paths
    model_path = (
        "/home/jupyter/DATA/model_weights/esm3_complete/esm3_sm_open_v1_state_dict.pt"
    )
    pdb_directory = (
        "/home/jupyter/DATA/hyperbind_train/sabdab/all_structures/processed/trimmed"
    )
    checkpoint_dir = "/home/jupyter/DATA/esm_finetuning/checkpoints"
    lora_checkpoints_dir = "/home/jupyter/DATA/esm_finetuning/new_lora_checkpoints_03_10"

    # Training parameters
    batch_size = 16  # Reduced to help with memory issues
    num_epochs = 5
    learning_rate = 5e-5  # Reduced to stabilize training
    weight_decay = 1e-6
    num_workers = 4

    # LoRA parameters
    lora_rank = 8  # Reduced rank for more stability
    lora_alpha = 16  # Reduced alpha to match rank
    lora_dropout = 0.05  # Lower dropout for stability

    # Ensure dirs exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(lora_checkpoints_dir, exist_ok=True)

    print(f"ESM version: {esm.__version__}")
    print("Using CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.get_device_name(0))

    # 1) Create base ESM-3 model, load pretrained weights
    model = ESM3_sm_open_v0()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 2) Freeze entire model so everything defaults to requires_grad=False
    for param in model.parameters():
        param.requires_grad = False

    # 3) Apply LoRA to strategic layers for structure prediction
    target_modules = [
        # Structure-focused output heads
        "output_heads.structure_head.0",  # First linear in structure head
        "output_heads.structure_head.3",  # Final projection in structure head

        # Structure decoder components
        "_structure_decoder.affine_output_projection.ffn1",
        "_structure_decoder.affine_output_projection.proj",
        "_structure_decoder.pairwise_classification_head.linear2",
        "_structure_decoder.plddt_head.dense",
        "_structure_decoder.plddt_head.output",

        # Late transformer blocks (higher-level reasoning)
        "transformer.blocks.45.attn.layernorm_qkv.1",
        "transformer.blocks.45.attn.out_proj",
        "transformer.blocks.46.attn.layernorm_qkv.1",
        "transformer.blocks.46.attn.out_proj",
        "transformer.blocks.47.attn.layernorm_qkv.1",
        "transformer.blocks.47.attn.out_proj",

        # Add some FFN components in critical blocks
        "transformer.blocks.45.ffn.1",
        "transformer.blocks.45.ffn.3",
        "transformer.blocks.47.ffn.1",
        "transformer.blocks.47.ffn.3",

        # Structure decoder transformer blocks
        "_structure_decoder.decoder_stack.blocks.28.attn.layernorm_qkv.1",
        "_structure_decoder.decoder_stack.blocks.28.attn.out_proj",
        "_structure_decoder.decoder_stack.blocks.29.attn.layernorm_qkv.1",
        "_structure_decoder.decoder_stack.blocks.29.attn.out_proj",
    ]

    print(f"Applying LoRA to {len(target_modules)} modules...")
    model, modified_modules = apply_lora_to_model(
        model,
        target_modules=target_modules,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )

    print(f"Successfully applied LoRA to {len(modified_modules)} modules:")
    for module_name in modified_modules:
        print(f"  - {module_name}")

    # Count LoRA trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} "
        f"({trainable_params / total_params:.2%} of total)"
    )

    # Move model to device
    model.to(device)

    # Create dataset with a validation strategy
    print("Creating and preparing dataset...")
    dataset = AntibodyDataset(pdb_directory)

    # Split dataset with more validation data for better evaluation
    val_fraction = 0.2  # 20% validation
    train_size = int((1 - val_fraction) * len(dataset))
    val_size = len(dataset) - train_size

    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Dataloaders with careful settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # Keep all examples
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Optimizer with specific settings for LoRA
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    decay_params = []
    no_decay_params = []

    for param in trainable_params:
        if len(param.shape) <= 1:  # Bias and LayerNorm parameters
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(param_groups, lr=learning_rate)

    # Loss function for structure prediction
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    # LR scheduler with patience and monitoring
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.6,
        patience=2,
        verbose=True,
        min_lr=1e-6,
    )

    # (Optional) Resume from a previous LoRA checkpoint
    resume_training = False
    resume_checkpoint = os.path.join(lora_checkpoints_dir, "lora_checkpoint_latest.pt")
    start_epoch = 0
    train_losses, val_losses = [], []

    if resume_training and os.path.exists(resume_checkpoint):
        print(f"Resuming training from {resume_checkpoint}")
        start_epoch, train_losses, val_losses = load_checkpoint(
            model, optimizer, scheduler, resume_checkpoint
        )

    # Train the model
    try:
        print("\n--- Starting Training with Structure-Focused LoRA ---\n")
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_fn,
            num_epochs,
            device,
            lora_checkpoints_dir,
            start_epoch,
            train_losses,
            val_losses,
        )

        # Plot and save final learning curves
        plot_learning_curves(
            train_losses,
            val_losses,
            os.path.join(lora_checkpoints_dir, "lora_learning_curves_final.png"),
        )

        # Save the final LoRA weights (just the fine-tuned parameters)
        final_lora_path = os.path.join(lora_checkpoints_dir, "lora_weights_final.pt")
        save_lora_weights(model, final_lora_path)

        # Also save a separate copy as latest for potential resuming
        latest_path = os.path.join(lora_checkpoints_dir, "lora_checkpoint_latest.pt")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            start_epoch + num_epochs,
            train_losses[-1],
            val_losses[-1],
            train_losses,
            val_losses,
            latest_path,
        )

        print("\n--- Training completed successfully ---\n")
        print(f"Final LoRA weights saved to: {final_lora_path}")
        print(f"Best validation loss: {min(val_losses):.4f}")

        # Quick verification of learned LoRA weights
        print("\nVerifying LoRA parameter statistics:")
        lora_norm_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_norm = torch.norm(param).item()
                if "lora_A" in name:
                    category = "lora_A"
                elif "lora_B" in name:
                    category = "lora_B"
                else:
                    category = "other"

                if category not in lora_norm_stats:
                    lora_norm_stats[category] = []
                lora_norm_stats[category].append(param_norm)

        for category, norms in lora_norm_stats.items():
            avg_norm = sum(norms) / len(norms)
            print(
                f"  {category}: avg norm = {avg_norm:.4f}, "
                f"min = {min(norms):.4f}, max = {max(norms):.4f}"
            )

    except Exception as exc:
        print(f"\nTraining error: {str(exc)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
