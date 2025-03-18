import os
import math
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple, Union

import esm
from esm.pretrained import ESM3_sm_open_v0
from esm.utils.encoding import tokenize_sequence

# Force synchronous CUDA execution (helpful for debugging)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Clear CPU/GPU caches
gc.collect()
torch.cuda.empty_cache()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


class AntibodyPairDataset(Dataset):
    """
    A dataset that loads paired antibody sequences and prepares them for fine-tuning.
    """

    def __init__(self, csv_file: str, tokenizer=None, max_length: int = 384):
        """
        Args:
            csv_file (str): Path to CSV file with paired antibody sequences.
            tokenizer: The ESM3 tokenizer to use.
            max_length (int): Maximum sequence length after tokenization.
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Check if we have the expected columns
        if not all(col in self.data.columns for col in ["sequence_alignment_aa_heavy", "sequence_alignment_aa_light"]):
            raise ValueError("CSV must contain 'sequence_alignment_aa_heavy' and 'sequence_alignment_aa_light' columns")
        
        # Define special tokens for ESM3
        self.pad_idx = 1      # Padding token
        self.mask_idx = 32    # Mask token for MLM
        self.cls_idx = 0      # Beginning of sequence
        self.eos_idx = 2      # End of sequence
        self.unk_idx = 3      # Unknown amino acid
        self.vocab_size = 64  # Standard vocabulary size for ESM3
        
        # Standard amino acid mapping for ESM3
        self.aa_to_idx = {
            'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11, 'H': 12, 'I': 13,
            'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18, 'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23,
            'B': 24, 'J': 25, 'O': 26, 'U': 27, 'X': 28, 'Z': 29, '.': 30, '-': 31, '|': 31  # Use '-' for separator
        }
        
        print(f"Loaded dataset with {len(self.data)} paired antibody sequences")

    def __len__(self) -> int:
        return len(self.data)
    
    def tokenize_antibody_pair(self, heavy_seq: str, light_seq: str) -> Tuple[List[int], List[int]]:
        """
        Tokenize a pair of heavy and light chain sequences.
        
        Args:
            heavy_seq (str): The heavy chain amino acid sequence.
            light_seq (str): The light chain amino acid sequence.
            
        Returns:
            Tuple containing token_ids and attention_mask
        """
        # Try using the ESM tokenizer if available
        if self.tokenizer and hasattr(self.tokenizer, 'batch_convert_sequences'):
            try:
                combined_seq = f"{heavy_seq}|{light_seq}"
                encoding = self.tokenizer.batch_convert_sequences([combined_seq])[0]
                return encoding["token_ids"], [1] * len(encoding["token_ids"])
            except Exception as e:
                print(f"Error using ESM tokenizer: {e}. Falling back to manual tokenization.")
        
        # Manual tokenization as fallback
        token_ids = [self.cls_idx]  # Start with <cls>
        
        # Tokenize heavy chain
        for aa in heavy_seq:
            token_ids.append(self.aa_to_idx.get(aa, self.unk_idx))
        
        # Add separator
        token_ids.append(self.aa_to_idx['|'])
        
        # Tokenize light chain
        for aa in light_seq:
            token_ids.append(self.aa_to_idx.get(aa, self.unk_idx))
        
        # Add end of sequence
        token_ids.append(self.eos_idx)
        
        # Create attention mask (1 for all tokens)
        attention_mask = [1] * len(token_ids)
        
        return token_ids, attention_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get heavy and light chain sequences
        heavy_seq = self.data.iloc[idx]["sequence_alignment_aa_heavy"]
        light_seq = self.data.iloc[idx]["sequence_alignment_aa_light"]
        
        # Tokenize
        token_ids, attention_mask = self.tokenize_antibody_pair(heavy_seq, light_seq)
        
        # Truncate if needed
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        # Pad sequences to max_length
        pad_length = self.max_length - len(token_ids)
        if pad_length > 0:
            token_ids = token_ids + [self.pad_idx] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Create labels for masked language modeling (MLM)
        labels = input_ids.clone()
        
        # Create mask excluding special tokens and padding
        # Only mask amino acids, not <cls>, <sep>, <eos>, <pad>
        special_tokens_mask = torch.tensor(
            [1 if (i in [self.cls_idx, self.eos_idx, self.pad_idx]) else 0 for i in token_ids],
            dtype=torch.bool
        )
        
        # Create a probability matrix for masking (15% of tokens)
        probability_matrix = torch.full(labels.shape, 0.15)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Sample tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels for unmasked tokens to -100 (ignored in loss)
        labels[~masked_indices] = -100
        
        # For masked indices:
        # - 80% replace with [MASK]
        # - 10% replace with random token
        # - 10% keep the same
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_idx
        
        # Replace 10% with random tokens
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(4, self.vocab_size-1, labels.shape, dtype=torch.long)  # Avoid special tokens
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # No need to pad as the dataset already handles padding
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels)
    }


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


def print_gpu_memory() -> None:
    """
    Print current GPU memory usage if CUDA is available.
    """
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory allocated: {allocated_gb:.2f} GB")
        print(f"GPU Memory reserved: {reserved_gb:.2f} GB")


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    """
    Evaluate model on a validation set.
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass (without labels parameter)
                outputs = model(sequence_tokens=input_ids)
                
                # Get the logits from the sequence head
                sequence_logits = outputs.sequence_logits
                
                # Calculate cross-entropy loss manually
                sequence_logits_view = sequence_logits.view(-1, sequence_logits.size(-1))
                labels_view = labels.view(-1)
                
                # Create loss function
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                
                # Calculate loss
                loss = loss_fct(sequence_logits_view, labels_view)
                
                total_loss += loss.item()
                valid_batches += 1

            except Exception as exc:
                print(f"Error in validation batch: {str(exc)}")
                continue

    if valid_batches > 0:
        return total_loss / valid_batches
    return float("inf")


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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[ReduceLROnPlateau],
    num_epochs: int,
    device: torch.device,
    checkpoint_dir: str,
    gradient_accumulation_steps: int = 1,
    start_epoch: int = 0,
    train_losses: Optional[List[float]] = None,
    val_losses: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
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
    patience = 5
    best_val_loss = float("inf")
    no_improve_count = 0

    # GradScaler for mixed precision training
    try:
        # Try newer syntax first
        scaler = torch.amp.GradScaler(device_type='cuda')
    except TypeError:
        # Fall back to older syntax
        scaler = torch.amp.GradScaler() if hasattr(torch.amp, 'GradScaler') else GradScaler()

    start_time = time.time()
    batch_times = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        valid_batches = 0

        # ----- Training Loop -----
        for batch_idx, batch in enumerate(train_loader):
            try:
                batch_start_time = time.time()

                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Mixed-precision forward
                try:
                    # Try newer syntax first with device_type
                    with torch.amp.autocast(device_type='cuda'):
                        # ESM3 model doesn't accept labels directly
                        outputs = model(sequence_tokens=input_ids)
                        
                        # Get the logits from the sequence head
                        sequence_logits = outputs.sequence_logits
                        
                        # Calculate cross-entropy loss manually
                        # Reshape logits to [batch_size * seq_length, vocab_size]
                        sequence_logits_view = sequence_logits.view(-1, sequence_logits.size(-1))
                        
                        # Reshape labels to [batch_size * seq_length]
                        labels_view = labels.view(-1)
                        
                        # Create loss function
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                        
                        # Calculate loss
                        loss = loss_fct(sequence_logits_view, labels_view)
                        
                        # Scale the loss if using gradient accumulation
                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps
                except TypeError:
                    try:
                        # Try just autocast() without arguments
                        with torch.amp.autocast():
                            # ESM3 model doesn't accept labels directly
                            outputs = model(sequence_tokens=input_ids)
                            
                            # Get the logits from the sequence head
                            sequence_logits = outputs.sequence_logits
                            
                            # Calculate cross-entropy loss manually
                            # Reshape logits to [batch_size * seq_length, vocab_size]
                            sequence_logits_view = sequence_logits.view(-1, sequence_logits.size(-1))
                            
                            # Reshape labels to [batch_size * seq_length]
                            labels_view = labels.view(-1)
                            
                            # Create loss function
                            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                            
                            # Calculate loss
                            loss = loss_fct(sequence_logits_view, labels_view)
                            
                            # Scale the loss if using gradient accumulation
                            if gradient_accumulation_steps > 1:
                                loss = loss / gradient_accumulation_steps
                    except Exception:
                        # Fall back to the old cuda.amp.autocast()
                        with autocast():
                            # ESM3 model doesn't accept labels directly
                            outputs = model(sequence_tokens=input_ids)
                            
                            # Get the logits from the sequence head
                            sequence_logits = outputs.sequence_logits
                            
                            # Calculate cross-entropy loss manually
                            # Reshape logits to [batch_size * seq_length, vocab_size]
                            sequence_logits_view = sequence_logits.view(-1, sequence_logits.size(-1))
                            
                            # Reshape labels to [batch_size * seq_length]
                            labels_view = labels.view(-1)
                            
                            # Create loss function
                            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                            
                            # Calculate loss
                            loss = loss_fct(sequence_logits_view, labels_view)
                            
                            # Scale the loss if using gradient accumulation
                            if gradient_accumulation_steps > 1:
                                loss = loss / gradient_accumulation_steps

                # Backward + gradient scale
                scaler.scale(loss).backward()
                
                # Track full loss
                total_loss += loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
                valid_batches += 1

                # Only update weights after accumulating enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

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
                        f"Loss: {loss.item() * gradient_accumulation_steps:.4f}, ETA: {eta}"
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
            val_loss = evaluate_model(model, val_loader, device)
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

            # Save checkpoint every epoch
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

            # Save LoRA weights separately (only trainable params)
            save_lora_weights(
                model,
                os.path.join(checkpoint_dir, f"lora_weights_epoch_{epoch + 1}.pt")
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
                save_lora_weights(
                    model,
                    os.path.join(checkpoint_dir, "best_lora_weights.pt")
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


def main():
    """
    Main function to prepare data, create model, apply LoRA, and start training.
    """
    # Configuration
    config = {
        # Paths
        "data_path": "/home/jupyter/DATA/hyperbind_train/oas/paired_oas_human_train.csv",
        "model_weights": "/home/jupyter/DATA/model_weights/esm3_complete/esm3_sm_open_v1_state_dict.pt",
        "output_dir": "./esm3_finetuned_antibody",
        
        # Training parameters
        "batch_size": 16,
        "gradient_accumulation_steps": 4,  # Effective batch size = 16 * 4 = 64
        "learning_rate": 2e-4,
        "weight_decay": 1e-6,
        "epochs": 3,
        "warmup_steps": 100,
        "max_seq_length": 384,  # Increased for antibody pairs
        "num_workers": 4,
        
        # LoRA parameters
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }
    
    # Ensure output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)
    
    print(f"ESM version: {esm.__version__}")
    print("Using CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.get_device_name(0))
    
    # 1) Create base ESM-3 model, load pretrained weights
    print("Loading ESM3 model...")
    model = ESM3_sm_open_v0()
    state_dict = torch.load(config["model_weights"], map_location=device)
    model.load_state_dict(state_dict)
    
    # Get tokenizer from model if possible
    try:
        from esm.tokenization import get_esm3_model_tokenizers
        from esm.utils.constants.models import ESM3_OPEN_SMALL
        tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
        sequence_tokenizer = tokenizers.sequence
        print("Successfully loaded ESM3 tokenizer")
    except Exception as e:
        print(f"Could not load ESM3 tokenizer: {e}")
        sequence_tokenizer = None
    
    # 2) Freeze entire model first
    for param in model.parameters():
        param.requires_grad = False
    
    # 3) Apply LoRA to strategic layers for antibody sequence modeling
    target_modules = [
        # Sequence-focused output head
        "output_heads.sequence_head.0",
        "output_heads.sequence_head.3",
        
        # Final transformer layers for high-level understanding
        "transformer.blocks.45.attn.layernorm_qkv.1",
        "transformer.blocks.45.attn.out_proj",
        "transformer.blocks.46.attn.layernorm_qkv.1",
        "transformer.blocks.46.attn.out_proj",
        "transformer.blocks.47.attn.layernorm_qkv.1",
        "transformer.blocks.47.attn.out_proj",
        
        # Feed-forward networks in final layers
        "transformer.blocks.45.ffn.1",
        "transformer.blocks.45.ffn.3",
        "transformer.blocks.46.ffn.1",
        "transformer.blocks.46.ffn.3",
        "transformer.blocks.47.ffn.1",
        "transformer.blocks.47.ffn.3",
        
        # Middle layers for complex relationships
        "transformer.blocks.35.attn.layernorm_qkv.1",
        "transformer.blocks.35.attn.out_proj",
        "transformer.blocks.40.attn.layernorm_qkv.1",
        "transformer.blocks.40.attn.out_proj",
    ]
    
    print(f"Applying LoRA to {len(target_modules)} modules...")
    model, modified_modules = apply_lora_to_model(
        model,
        target_modules=target_modules,
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
        dropout=config["lora_dropout"],
    )
    
    print(f"Successfully applied LoRA to {len(modified_modules)} modules:")
    for i, module_name in enumerate(modified_modules):
        if i < 10 or i >= len(modified_modules) - 5:  # Print first 10 and last 5
            print(f"  - {module_name}")
        elif i == 10:
            print(f"  - ... {len(modified_modules) - 15} more modules ...")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    # Move model to device
    model.to(device)
    
    # Create dataset and dataloader
    print(f"Loading antibody dataset from {config['data_path']}...")
    dataset = AntibodyPairDataset(
        csv_file=config['data_path'],
        tokenizer=sequence_tokenizer,
        max_length=config['max_seq_length']
    )
    
    # Split dataset for validation
    val_fraction = 0.1  # 10% for validation
    train_size = int((1 - val_fraction) * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    
    # Optimizer with weight decay differentiation
    # Separate parameters with weight decay from those without
    decay_params = []
    no_decay_params = []
    
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) <= 1:  # Bias and LayerNorm parameters
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": config["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True,
        min_lr=1e-6
    )
    
    # Train the model
    print("\n----- Starting Training with LoRA for Antibody Sequences -----\n")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config["epochs"],
        device=device,
        checkpoint_dir=config["output_dir"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"]
    )
    
    # Save final model and LoRA weights
    final_model_path = os.path.join(config["output_dir"], "final_model.pt")
    final_lora_path = os.path.join(config["output_dir"], "final_lora_weights.pt")
    
    # Save full model
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Save just the LoRA weights
    save_lora_weights(model, final_lora_path)
    print(f"Final LoRA weights saved to: {final_lora_path}")
    
    # Plot final learning curves
    plot_learning_curves(
        train_losses,
        val_losses,
        os.path.join(config["output_dir"], "learning_curves_final.png")
    )
    
    print("\n----- Training completed successfully -----\n")
    print(f"Best validation loss: {min(val_losses):.4f}")


if __name__ == "__main__":
    main()
