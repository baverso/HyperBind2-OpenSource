import os
import glob
import gc
import time
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt

from Bio import PDB
from Bio.PDB import *

# Force synchronous CUDA execution
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Clear caches
gc.collect()
torch.cuda.empty_cache()

# Import ESM modules
import esm
from esm.pretrained import ESM3_sm_open_v0, ESM3_structure_encoder_v0
from esm.tokenization import get_esm3_model_tokenizers
from esm.utils.encoding import tokenize_sequence, tokenize_structure
from esm.utils.constants.models import ESM3_OPEN_SMALL


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get tokenizers
tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
sequence_tokenizer = tokenizers.sequence
structure_tokenizer = tokenizers.structure

# Initialize structure encoder on CPU
structure_encoder = ESM3_structure_encoder_v0(device="cpu")


class StructureData:
    """
    Helper class to ensure proper coordinate handling.
    """

    def __init__(self, sequence, atom_coords, atom_mask=None):
        self.sequence = sequence
        # coords should be [n_res, n_atoms, 3]
        self.atom_coords = atom_coords
        self.atom_mask = (
            atom_mask
            if atom_mask is not None
            else torch.ones_like(atom_coords[..., 0], dtype=torch.bool)
        )


def process_pdb_file(file_path):
    """
    Process a PDB file and return sequence and coordinates.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)

    # Three to one letter conversion dictionary
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

    # Atoms to extract (in order)
    atoms_to_extract = ["N", "CA", "C", "O", "CB"]

    # First collect all valid residues
    valid_residues = []
    sequence = []

    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if it's a valid amino acid and has at least CA
                if "CA" in residue and residue.get_resname() in three_to_one:
                    valid_residues.append(residue)
                    sequence.append(three_to_one[residue.get_resname()])

    if not valid_residues:
        return None

    sequence = "".join(sequence)
    n_res = len(sequence)
    n_atoms = len(atoms_to_extract)

    # Initialize coordinate tensor [n_res, n_atoms, 3]
    coords = np.zeros((n_res, n_atoms, 3))
    atom_mask = np.zeros((n_res, n_atoms), dtype=bool)

    # Fill in coordinates and mask
    for i, residue in enumerate(valid_residues):
        for j, atom_name in enumerate(atoms_to_extract):
            if atom_name in residue:
                coords[i, j] = residue[atom_name].get_coord()
                atom_mask[i, j] = True
            elif atom_name == "CB" and residue.get_resname() == "GLY":
                # Handle missing CB in Glycine
                # (Could compute virtual CB here if needed)
                continue

    # Convert to tensors
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    mask_tensor = torch.tensor(atom_mask, dtype=torch.bool)

    return StructureData(
        sequence=sequence, atom_coords=coords_tensor, atom_mask=mask_tensor
    )


class AntibodyDataset(Dataset):
    def __init__(self, pdb_dir):
        self.pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        print(f"Found {len(self.pdb_files)} PDB files")

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        try:
            pdb_file = self.pdb_files[idx]
            structure_data = process_pdb_file(pdb_file)
            if structure_data is None:
                print(f"Failed to process {pdb_file}")
                return None

            # Tokenize sequence
            try:
                input_seq = tokenize_sequence(
                    structure_data.sequence, sequence_tokenizer, add_special_tokens=True
                )
            except Exception as e:
                print(f"Sequence tokenization failed for {pdb_file}: {str(e)}")
                return None

            # Tokenize structure
            try:
                # Get structure tokens
                token_logits, token_probs, structure_tokens = tokenize_structure(
                    structure_data.atom_coords,
                    structure_encoder,
                    structure_tokenizer,
                    reference_sequence=structure_data.sequence,
                    add_special_tokens=True,
                )

                # Clamp token values to valid range (0 to 4095, -100 for padding)
                valid_tokens = structure_tokens != -100
                structure_tokens[valid_tokens] = torch.clamp(
                    structure_tokens[valid_tokens], 0, 4095
                )

            except Exception as e:
                print(f"Structure tokenization failed for {pdb_file}: {str(e)}")
                import traceback

                traceback.print_exc()
                return None

            return {
                "sequence": input_seq,
                "structure": structure_tokens,
                "file": pdb_file,
            }

        except Exception as e:
            print(f"Error processing {pdb_file}: {str(e)}")
            return None


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        print("No valid items in batch")
        return None

    try:
        sequences = [item["sequence"] for item in batch]
        structures = [item["structure"] for item in batch]
        files = [item["file"] for item in batch]

        # Pad sequences and structures
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0
        )
        padded_structures = torch.nn.utils.rnn.pad_sequence(
            structures, batch_first=True, padding_value=-100
        )

        return {
            "sequence": padded_sequences,
            "structure": padded_structures,
            "files": files,
        }

    except Exception as e:
        print(f"Error in collate_fn: {str(e)}")
        return None


def print_gpu_memory():
    """
    Print current GPU memory usage.
    """
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    val_loss,
    train_losses,
    val_losses,
    filename,
):
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


def load_checkpoint(model, optimizer, scheduler, filename):
    """
    Load model checkpoint and training state.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"]
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], train_losses, val_losses


def evaluate_model(model, dataloader, loss_fn, device):
    """
    Evaluate model on validation set.
    """
    model.eval()
    total_loss = 0
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            try:
                # Move tensors to device
                input_seq = batch["sequence"].to(device)
                target_structure = batch["structure"].to(device)

                # Forward pass
                output = model(
                    sequence_tokens=input_seq, structure_tokens=target_structure
                )
                pred_logits = output.structure_logits

                # Ensure target values are within valid range
                n_classes = pred_logits.size(-1)
                if target_structure.max() >= n_classes:
                    print("Warning: target values exceed number of classes")
                    continue

                # Compute loss
                loss = loss_fn(
                    pred_logits.view(-1, n_classes), target_structure.view(-1)
                )

                total_loss += loss.item()
                valid_batches += 1

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float("inf")
    return avg_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    num_epochs,
    device,
    checkpoint_dir,
    start_epoch=0,
    train_losses=None,
    val_losses=None,
):
    """
    Train model with validation and checkpointing.
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

    # Gradient scaler for mixed precision
    scaler = GradScaler()

    # Track time for ETA calculation
    start_time = time.time()
    batch_times = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        valid_batches = 0

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            try:
                batch_start_time = time.time()

                # Move tensors to device
                input_seq = batch["sequence"].to(device)
                target_structure = batch["structure"].to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass with mixed precision
                with autocast():
                    output = model(
                        sequence_tokens=input_seq, structure_tokens=target_structure
                    )
                    pred_logits = output.structure_logits

                    # Ensure target values are within valid range
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

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights with scaler
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                valid_batches += 1

                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)

                # ETA calculation
                if len(batch_times) > 10:
                    avg_batch_time = sum(batch_times[-10:]) / 10
                    remaining_batches = len(train_loader) - (batch_idx + 1)
                    remaining_time = avg_batch_time * remaining_batches
                    eta = datetime.fromtimestamp(time.time() + remaining_time).strftime(
                        "%H:%M:%S"
                    )
                else:
                    eta = "calculating..."

                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(
                        f"Epoch {epoch+1}/{start_epoch + num_epochs}, "
                        f"Batch {batch_idx+1}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}, ETA: {eta}"
                    )
                    print_gpu_memory()

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if valid_batches > 0:
            avg_train_loss = total_loss / valid_batches
            train_losses.append(avg_train_loss)

            print("\nValidating model...")
            val_loss = evaluate_model(model, val_loader, loss_fn, device)
            val_losses.append(val_loss)

            # Update scheduler
            if scheduler:
                scheduler.step(val_loss)

            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Plot and save learning curves
            plot_learning_curves(
                train_losses,
                val_losses,
                os.path.join(checkpoint_dir, f"learning_curves_epoch_{epoch+1}.png"),
            )

            # Save regular checkpoint
            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    avg_train_loss,
                    val_loss,
                    train_losses,
                    val_losses,
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                )

            # Check for improvement and save best model
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
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            print(f"Epoch {epoch+1} completed with no valid batches")

    total_training_time = time.time() - start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    return train_losses, val_losses


def plot_learning_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation learning curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    min_train_epoch = train_losses.index(min_train_loss) + 1
    min_val_epoch = val_losses.index(min_val_loss) + 1

    plt.text(
        0.02,
        0.95,
        f"Min Train Loss: {min_train_loss:.4f} (Epoch {min_train_epoch})",
        transform=plt.gca().transAxes,
        fontsize=9,
    )
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


def main():
    """
    Main function to set up data, model, and training loop.
    """
    # Set paths
    model_path = (
        "/home/jupyter/DATA/model_weights/esm3_complete/esm3_sm_open_v1_state_dict.pt"
    )
    pdb_directory = (
        "/home/jupyter/DATA/hyperbind_train/sabdab/all_structures/processed/trimmed"
    )
    checkpoint_dir = "/home/jupyter/DATA/esm_finetuning/checkpoints"

    # Set training parameters
    batch_size = 16  # Increased from 4
    num_epochs = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_workers = 4

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"ESM version: {esm.__version__}")
    print("Using CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.get_device_name(0))

    # Create model
    model = ESM3_sm_open_v0()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Create dataset
    dataset = AntibodyDataset(pdb_directory)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Setup training
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Check for resume checkpoint
    resume_training = False  # Set to True to resume from a checkpoint
    resume_checkpoint = os.path.join(checkpoint_dir, "checkpoint_epoch_20.pt")
    start_epoch = 0
    train_losses = []
    val_losses = []

    if resume_training and os.path.exists(resume_checkpoint):
        start_epoch, train_losses, val_losses = load_checkpoint(
            model, optimizer, scheduler, resume_checkpoint
        )

    # Train model
    try:
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_fn,
            num_epochs,
            device,
            checkpoint_dir,
            start_epoch,
            train_losses,
            val_losses,
        )
        plot_learning_curves(
            train_losses,
            val_losses,
            os.path.join(checkpoint_dir, "final_learning_curves.png"),
        )
        print("Training completed successfully")
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

