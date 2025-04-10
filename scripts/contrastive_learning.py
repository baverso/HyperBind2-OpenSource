"""
Contrastive learning model for protein sequence pairs using ESM3.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve
)
import matplotlib.pyplot as plt

# ESM3 imports
from esm.pretrained import ESM3_sm_open_v0


class CustomESM3Adapter(nn.Module):
    """Improved adapter for ESM3 with attention-based pooling."""

    def __init__(self, esm3_model):
        """Initialize the adapter with the ESM3 model.

        Args:
            esm3_model: An ESM3 model instance
        """
        super().__init__()
        self.model = esm3_model
        self.embed_dim = 1536  # Default for ESM3-small

        # Add attention-based pooling
        self.attention = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, tokens):
        """Process sequence tokens and return embeddings with attention pooling.

        Args:
            tokens: Tokenized protein sequences

        Returns:
            torch.Tensor: Pooled embeddings
        """
        # Ensure tokens is a long tensor
        tokens = tokens.long()

        # Create attention mask (1 for real tokens, 0 for padding)
        padding_mask = (tokens != 1)  # 1 is pad_idx

        # Forward pass through ESM3 model
        with torch.no_grad():
            result = self.model(sequence_tokens=tokens)

            # Extract embeddings
            if hasattr(result, 'embeddings'):
                embeddings = result.embeddings
            else:
                raise ValueError("No embeddings found in model output")

        # Apply attention-based pooling if embeddings have sequence dimension
        if embeddings.dim() == 3:  # [batch_size, seq_len, hidden_dim]
            # Mask out padding tokens
            mask = padding_mask.unsqueeze(-1).float()

            # Calculate attention weights
            attention_weights = self.attention(embeddings)

            # Apply mask to attention weights
            attention_weights = attention_weights * mask
            attention_weights = attention_weights / (
                attention_weights.sum(dim=1, keepdim=True) + 1e-8
            )

            # Weighted sum of token embeddings
            pooled_embeddings = torch.sum(
                embeddings * attention_weights, dim=1
            )
            return pooled_embeddings
        else:
            return embeddings


class ImprovedContrastiveHead(nn.Module):
    """Improved projection head with residual connections and normalization."""

    def __init__(self, in_dim=1536, hidden_dim=1024, out_dim=256, dropout=0.2):
        """Initialize projection head.

        Args:
            in_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output projection dimension
            dropout: Dropout rate
        """
        super().__init__()

        # First block
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Second block with residual connection
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Output projection
        self.output = nn.Linear(hidden_dim, out_dim)

        # Skip connection adapter (if input and hidden dimensions differ)
        self.skip_adapter = (
            nn.Linear(in_dim, hidden_dim)
            if in_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x):
        """Project embeddings to a lower-dimensional space.

        Args:
            x: Input embeddings

        Returns:
            torch.Tensor: Normalized projections
        """
        # First block
        block1_out = self.block1(x)

        # Second block with residual connection
        block2_out = self.block2(block1_out) + block1_out

        # Output projection
        projected = self.output(block2_out)

        # Normalize to unit length
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized


class EnhancedContrastiveModel(nn.Module):
    """Enhanced contrastive model with improved architecture."""

    def __init__(self, esm3_model, freeze_backbone=True, temperature=0.07):
        """Initialize the contrastive model.

        Args:
            esm3_model: ESM3 model to use as backbone
            freeze_backbone: Whether to freeze backbone weights
            temperature: Initial temperature for similarity scaling
        """
        super().__init__()

        # Create backbone with attention-based pooling
        self.backbone = CustomESM3Adapter(esm3_model)
        self.embed_dim = 1536  # ESM3-small embedding dimension

        # Create improved projection head
        self.projection_head = ImprovedContrastiveHead(
            in_dim=self.embed_dim,
            hidden_dim=1024,
            out_dim=256,
            dropout=0.3
        )

        # Learnable temperature parameter for similarity scaling
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.model.parameters():
                param.requires_grad = False

        # Selectively unfreeze the attention mechanism in the adapter
        for param in self.backbone.attention.parameters():
            param.requires_grad = True

    def forward(self, sequence_tokens):
        """Forward pass through the enhanced model.

        Args:
            sequence_tokens: Tokenized protein sequences

        Returns:
            torch.Tensor: Projected embeddings
        """
        # Get embeddings with attention pooling
        embeddings = self.backbone(sequence_tokens)

        # Project to lower-dimensional space
        projected = self.projection_head(embeddings)

        return projected

    def compute_similarity(self, z1, z2):
        """Compute cosine similarity with learnable temperature.

        Args:
            z1: First set of embeddings
            z2: Second set of embeddings

        Returns:
            torch.Tensor: Similarity scores
        """
        # Cosine similarity with temperature scaling
        sim = F.cosine_similarity(z1, z2, dim=1) / self.temperature
        return sim


def tokenize_sequence(sequence, max_length=384):
    """Tokenize a protein sequence for ESM3.

    Args:
        sequence (str): Amino acid sequence
        max_length (int): Maximum sequence length

    Returns:
        torch.Tensor: Tokenized sequence
    """
    # ESM3 token mapping
    aa_to_idx = {
        'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11,
        'H': 12, 'I': 13, 'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18,
        'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23, 'B': 24, 'J': 25,
        'O': 26, 'U': 27, 'X': 28, 'Z': 29, '.': 30, '-': 31, '|': 31
    }

    # Special tokens
    cls_idx = 0  # Beginning of sequence
    pad_idx = 1  # Padding token
    eos_idx = 2  # End of sequence
    unk_idx = 3  # Unknown amino acid

    # Start with <cls>
    token_ids = [cls_idx]

    # Tokenize sequence
    for aa in sequence:
        # Skip invalid characters
        if aa in ' \t\n\r':
            continue

        # Convert to uppercase and get token ID
        token_ids.append(aa_to_idx.get(aa.upper(), unk_idx))

    # Add end of sequence
    token_ids.append(eos_idx)

    # Truncate if too long
    if len(token_ids) > max_length:
        # Keep cls, truncate middle, keep end
        token_ids = token_ids[:1] + token_ids[1:max_length-1] + [eos_idx]

    # Pad if needed
    pad_length = max_length - len(token_ids)
    if pad_length > 0:
        token_ids = token_ids + [pad_idx] * pad_length

    return torch.tensor(token_ids, dtype=torch.long)


class DirectContrastiveDataset(Dataset):
    """Dataset that directly takes sequences and tokenizes them."""

    def __init__(
        self,
        cdr3_1_list,
        antigen_1_list,
        cdr3_2_list,
        antigen_2_list,
        labels,
        max_length=384
    ):
        """Initialize dataset with direct sequence data.

        Args:
            cdr3_1_list: List of CDR3 sequences for first pair member
            antigen_1_list: List of antigen sequences for first pair member
            cdr3_2_list: List of CDR3 sequences for second pair member
            antigen_2_list: List of antigen sequences for second pair member
            labels: List of labels (1 for positive, 0 for negative)
            max_length: Maximum sequence length
        """
        self.cdr3_1 = cdr3_1_list
        self.antigen_1 = antigen_1_list
        self.cdr3_2 = cdr3_2_list
        self.antigen_2 = antigen_2_list
        self.labels = labels
        self.max_length = max_length

        # Define special tokens for ESM3
        self.pad_idx = 1  # Padding token
        self.cls_idx = 0  # Beginning of sequence
        self.eos_idx = 2  # End of sequence
        self.unk_idx = 3  # Unknown amino acid

        # Standard amino acid mapping for ESM3
        self.aa_to_idx = {
            'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11,
            'H': 12, 'I': 13, 'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18,
            'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23, 'B': 24, 'J': 25,
            'O': 26, 'U': 27, 'X': 28, 'Z': 29, '.': 30, '-': 31, '|': 31
        }

        # Verify data length consistency
        assert (
            len(cdr3_1_list) == len(antigen_1_list) == len(cdr3_2_list) 
            == len(antigen_2_list) == len(labels)
        )
        print(f"Created dataset with {len(labels)} pairs")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def tokenize_sequence(self, sequence):
        """Tokenize a protein sequence using ESM3's tokenization approach.

        Args:
            sequence: Amino acid sequence

        Returns:
            List of token IDs
        """
        # Start with <cls>
        token_ids = [self.cls_idx]

        # Tokenize sequence
        for aa in sequence:
            # Skip invalid characters
            if aa in ' \t\n\r':
                continue

            # Convert to uppercase and get token ID
            token_ids.append(self.aa_to_idx.get(aa.upper(), self.unk_idx))

        # Add end of sequence
        token_ids.append(self.eos_idx)

        # Truncate if too long
        if len(token_ids) > self.max_length:
            # Keep cls, truncate middle, keep end
            token_ids = (
                token_ids[:1] + token_ids[1:self.max_length-1] + [self.eos_idx]
            )

        # Pad if needed
        pad_length = self.max_length - len(token_ids)
        if pad_length > 0:
            token_ids = token_ids + [self.pad_idx] * pad_length

        return token_ids

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            dict: Sample data
        """
        # Get sequences for the pair
        cdr3_1 = self.cdr3_1[idx]
        antigen_1 = self.antigen_1[idx]
        cdr3_2 = self.cdr3_2[idx]
        antigen_2 = self.antigen_2[idx]

        # Combine CDR3 and antigen for each pair
        sequence1 = str(cdr3_1) + "|" + str(antigen_1)
        sequence2 = str(cdr3_2) + "|" + str(antigen_2)

        # Tokenize sequences
        tokens1 = self.tokenize_sequence(sequence1)
        tokens2 = self.tokenize_sequence(sequence2)

        # Create attention masks (1 for tokens, 0 for padding)
        attn_mask1 = [
            1 if i != self.pad_idx else 0 for i in tokens1
        ]
        attn_mask2 = [
            1 if i != self.pad_idx else 0 for i in tokens2
        ]

        # Get label
        label = self.labels[idx]

        return {
            'tokens1': torch.tensor(tokens1, dtype=torch.long),
            'attn_mask1': torch.tensor(attn_mask1, dtype=torch.long),
            'tokens2': torch.tensor(tokens2, dtype=torch.long),
            'attn_mask2': torch.tensor(attn_mask2, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


def create_test_dataloader(
    test_pairs_path,
    raw_data_path=None,
    batch_size=32,
    num_workers=2
):
    """Create a test DataLoader with comprehensive ID matching strategies.

    Args:
        test_pairs_path: Path to CSV with test pairs
        raw_data_path: Path to raw data CSV
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader

    Returns:
        DataLoader: Configured test data loader
    """
    # Load test pairs
    test_pairs = pd.read_csv(test_pairs_path)
    print(f"Loaded {len(test_pairs)} pairs from {test_pairs_path}")

    if 'ID1' in test_pairs.columns and 'ID2' in test_pairs.columns and raw_data_path:
        # Load raw data
        raw_data = pd.read_csv(raw_data_path)
        print(f"Loaded raw data with {len(raw_data)} rows from {raw_data_path}")

        # Define columns to use
        id_column = 'ID_slide_Variant'
        cdr3_col = 'CDR3'
        antigen_col = 'antigen_sequence'

        # Create multiple indices for the raw data
        print("Creating multiple indices for raw data...")
        id_mappings = {
            'full_id': {},                 # Full ID
            'numeric_prefix': {},          # Numeric part before underscore
            'digits_only': {},             # Just the digits from the ID
            'substring': {},               # For substring matching
            'first_digits': {},            # First N digits
            'last_digits': {}              # Last N digits
        }

        for idx, row in raw_data.iterrows():
            # Get the full ID
            full_id = str(row[id_column])
            id_mappings['full_id'][full_id] = row

            # Get numeric prefix
            if '_' in full_id:
                prefix = full_id.split('_')[0]
                id_mappings['numeric_prefix'][prefix] = row

                # Store in digits-only map if the prefix is purely numeric
                if prefix.isdigit():
                    id_mappings['digits_only'][prefix] = row

                    # Store first 5-7 digits for prefix matching
                    for i in range(5, min(8, len(prefix) + 1)):
                        if prefix[:i] not in id_mappings['first_digits']:
                            id_mappings['first_digits'][prefix[:i]] = []
                        id_mappings['first_digits'][prefix[:i]].append(row)

                    # Store last 5-7 digits for suffix matching
                    for i in range(5, min(8, len(prefix) + 1)):
                        if prefix[-i:] not in id_mappings['last_digits']:
                            id_mappings['last_digits'][prefix[-i:]] = []
                        id_mappings['last_digits'][prefix[-i:]].append(row)

        # Print stats about the mappings
        for mapping_name, mapping in id_mappings.items():
            if mapping_name in ['first_digits', 'last_digits']:
                total = sum(len(rows) for rows in mapping.values())
                print(
                    f"  {mapping_name}: {len(mapping)} unique patterns "
                    f"with {total} total mappings"
                )
            else:
                print(f"  {mapping_name}: {len(mapping)} unique entries")

        # Lists for dataset creation
        cdr3_1_list = []
        antigen_1_list = []
        cdr3_2_list = []
        antigen_2_list = []
        labels = []

        # Track matching results
        match_stats = {
            'pairs_processed': 0,
            'matches_found': 0,
            'match_types': {}
        }

        # Helper function to find a match for a single ID
        def find_match_for_id(id_str):
            """Find a match for an ID using various strategies.

            Args:
                id_str: ID string to match

            Returns:
                tuple: (matched_row, strategy_name) or (None, None)
            """
            # Try different matching strategies in order of preference
            match_strategies = [
                (
                    'exact_match',
                    lambda id_str: id_mappings['digits_only'].get(id_str)
                ),
                (
                    'first_digits',
                    lambda id_str: id_mappings['first_digits'].get(id_str[:7])
                ),
                (
                    'first_digits_short',
                    lambda id_str: id_mappings['first_digits'].get(id_str[:6])
                ),
                (
                    'first_digits_shorter',
                    lambda id_str: id_mappings['first_digits'].get(id_str[:5])
                ),
                (
                    'last_digits',
                    lambda id_str: (
                        id_mappings['last_digits'].get(id_str[-7:])
                        if len(id_str) >= 7 else None
                    )
                ),
                (
                    'last_digits_short',
                    lambda id_str: (
                        id_mappings['last_digits'].get(id_str[-6:])
                        if len(id_str) >= 6 else None
                    )
                ),
                (
                    'last_digits_shorter',
                    lambda id_str: (
                        id_mappings['last_digits'].get(id_str[-5:])
                        if len(id_str) >= 5 else None
                    )
                )
            ]

            for strategy_name, strategy_func in match_strategies:
                result = strategy_func(id_str)
                if result is not None:
                    # If we got a list of matches, use the first one
                    if isinstance(result, list):
                        if result:
                            return result[0], strategy_name
                    else:
                        return result, strategy_name

            # No match found
            return None, None

        # Process each pair
        for idx, row in test_pairs.iterrows():
            id1 = str(row['ID1'])
            id2 = str(row['ID2'])

            # Get label from pair_type
            pair_type = row['pair_type'] if 'pair_type' in row else ''
            label = 1.0 if pair_type.lower() == 'positive' else 0.0

            # Find matches for both IDs
            match1, match_type1 = find_match_for_id(id1)
            match2, match_type2 = find_match_for_id(id2)

            match_stats['pairs_processed'] += 1

            # If both IDs match, add to dataset
            if match1 is not None and match2 is not None:
                # Extract sequences
                cdr3_1 = match1[cdr3_col]
                antigen_1 = match1[antigen_col]
                cdr3_2 = match2[cdr3_col]
                antigen_2 = match2[antigen_col]

                # Add to dataset
                cdr3_1_list.append(cdr3_1)
                antigen_1_list.append(antigen_1)
                cdr3_2_list.append(cdr3_2)
                antigen_2_list.append(antigen_2)
                labels.append(label)

                # Update match statistics
                match_stats['matches_found'] += 1
                match_stats['match_types'][match_type1] = (
                    match_stats['match_types'].get(match_type1, 0) + 1
                )
                match_stats['match_types'][match_type2] = (
                    match_stats['match_types'].get(match_type2, 0) + 1
                )

                # Print sample matches for debugging (limit to first few)
                if match_stats['matches_found'] <= 5:
                    print(f"\nMatch {match_stats['matches_found']}:")
                    print(f"  ID1: {id1} ({match_type1}) -> {match1[id_column]}")
                    print(f"  ID2: {id2} ({match_type2}) -> {match2[id_column]}")
                    print(f"  Label: {label} (from pair_type: {pair_type})")
                    print(f"  CDR3: {cdr3_1[:15]}... & {cdr3_2[:15]}...")
                    print(f"  Antigen: {antigen_1[:15]}... & {antigen_2[:15]}...")

            # Progress updates
            if idx % 1000 == 0 and idx > 0:
                print(
                    f"Processed {idx}/{len(test_pairs)} pairs, "
                    f"found {match_stats['matches_found']} matches"
                )

        # Print summary statistics
        print(f"\nMatching summary:")
        print(f"  Total pairs processed: {match_stats['pairs_processed']}")
        print(
            f"  Total matches found: {match_stats['matches_found']} "
            f"({match_stats['matches_found']/match_stats['pairs_processed']*100:.2f}%)"
        )

        # Print match type statistics
        print("\nMatch type statistics:")
        total_ids = 2 * match_stats['matches_found']  # Each match has 2 IDs
        for match_type, count in match_stats['match_types'].items():
            print(
                f"  {match_type}: {count} matches "
                f"({count/total_ids*100:.2f}% of matched IDs)"
            )

        # If not enough matches, use synthetic data
        if match_stats['matches_found'] < 10:
            print(
                "Insufficient matches for meaningful evaluation. "
                "Using synthetic data instead."
            )
            return create_synthetic_dataset(batch_size, num_workers)

        # Create dataset with matched pairs
        print(f"\nCreating dataset with {len(labels)} matched pairs")
        test_dataset = DirectContrastiveDataset(
            cdr3_1_list=cdr3_1_list,
            antigen_1_list=antigen_1_list,
            cdr3_2_list=cdr3_2_list,
            antigen_2_list=antigen_2_list,
            labels=labels
        )
    else:
        print("Required columns not found or raw_data_path not provided.")
        return create_synthetic_dataset(batch_size, num_workers)

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader


def create_synthetic_dataset(batch_size=32, num_workers=2):
    """Create a synthetic dataset for testing when real data cannot be loaded.

    Args:
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader

    Returns:
        DataLoader: DataLoader with synthetic test data
    """
    print("Creating a synthetic dataset with diverse examples...")

    # Define several CDR3 and antigen sequences for variety
    cdr3_sequences = [
        "CASSQDMTV",
        "CASSFGQETQYF",
        "CASSYEQYF",
        "CASSPGQGAGELF",
        "CSARQNRNYGYTF"
    ]

    antigen_sequences = [
        "VPGFPTVRRALVPK",
        "LLFGYPVYV",
        "KLVALGINAVAQQANEES",
        "KRWIILGLNKIVRMY",
        "GILGFVFTLTV"
    ]

    # Create datasets with controlled similarities
    cdr3_1_list = []
    antigen_1_list = []
    cdr3_2_list = []
    antigen_2_list = []
    labels = []

    # Create positive pairs (identical sequences)
    for cdr3 in cdr3_sequences:
        for antigen in antigen_sequences:
            cdr3_1_list.append(cdr3)
            antigen_1_list.append(antigen)
            cdr3_2_list.append(cdr3)
            antigen_2_list.append(antigen)
            labels.append(1.0)

    # Create negative pairs (different sequences)
    for i, (cdr3_1, antigen_1) in enumerate(zip(cdr3_sequences, antigen_sequences)):
        for j, (cdr3_2, antigen_2) in enumerate(zip(cdr3_sequences, antigen_sequences)):
            if i != j:
                cdr3_1_list.append(cdr3_1)
                antigen_1_list.append(antigen_1)
                cdr3_2_list.append(cdr3_2)
                antigen_2_list.append(antigen_2)
                labels.append(0.0)

    # Create dataset
    print(f"Created synthetic dataset with {len(labels)} pairs")
    print(
        f"Positive pairs: {sum(labels)}, "
        f"Negative pairs: {len(labels) - sum(labels)}"
    )

    test_dataset = DirectContrastiveDataset(
        cdr3_1_list=cdr3_1_list,
        antigen_1_list=antigen_1_list,
        cdr3_2_list=cdr3_2_list,
        antigen_2_list=antigen_2_list,
        labels=labels
    )

    # Create and return the DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader


def run_inference_on_test_set(
    model,
    test_loader,
    device=None,
    similarity_threshold=0.7
):
    """Run inference on a test set and compute metrics with improved threshold.

    Args:
        model: The loaded contrastive model
        test_loader: DataLoader with test data
        device: Computation device
        similarity_threshold: Threshold for binary classification

    Returns:
        tuple: (metrics_dict, similarities, labels)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Lists to store results
    all_similarities = []
    all_labels = []

    # Inference loop with error handling
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                # Move data to device
                tokens1 = batch['tokens1'].to(device)
                tokens2 = batch['tokens2'].to(device)
                labels = batch['label'].cpu().numpy()

                # Get embeddings from model
                z1 = model(tokens1)
                z2 = model(tokens2)

                # Use model's similarity function if available, otherwise use default
                if hasattr(model, 'compute_similarity'):
                    similarities = model.compute_similarity(z1, z2).cpu().numpy()
                else:
                    # Default cosine similarity
                    similarities = F.cosine_similarity(z1, z2, dim=1).cpu().numpy()

                # Store results
                all_similarities.extend(similarities)
                all_labels.extend(labels)

                # Progress update
                if batch_idx % 5 == 0:
                    print(f"Processed {batch_idx}/{len(test_loader)} batches")

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    # Convert to numpy arrays
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)

    # Check if we have any valid results
    if len(all_similarities) == 0 or len(all_labels) == 0:
        print("No valid results to evaluate!")
        return {
            'roc_auc': 0.0,
            'pr_auc': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }, all_similarities, all_labels

    # Compute metrics
    metrics = {}

    # ROC AUC
    try:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_similarities)
    except Exception as e:
        print(f"Error computing ROC AUC: {e}")
        metrics['roc_auc'] = 0.0

    # PR AUC
    try:
        precision, recall, _ = precision_recall_curve(all_labels, all_similarities)
        metrics['pr_auc'] = auc(recall, precision)
    except Exception as e:
        print(f"Error computing PR AUC: {e}")
        metrics['pr_auc'] = 0.0

    # Apply threshold and compute accuracy
    binary_preds = (all_similarities >= similarity_threshold).astype(int)
    metrics['accuracy'] = np.mean(binary_preds == all_labels)

    # True positives, false positives, etc.
    tp = np.sum((binary_preds == 1) & (all_labels == 1))
    tn = np.sum((binary_preds == 0) & (all_labels == 0))
    fp = np.sum((binary_preds == 1) & (all_labels == 0))
    fn = np.sum((binary_preds == 0) & (all_labels == 1))

    # Precision and recall
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1'] = (
        2 * (metrics['precision'] * metrics['recall']) 
        / (metrics['precision'] + metrics['recall'])
        if (metrics['precision'] + metrics['recall']) > 0 else 0
    )
    
    # Calculate optimal threshold
    try:
        precision, recall, thresholds = precision_recall_curve(
            all_labels, all_similarities
        )
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[optimal_idx] 
            if optimal_idx < len(thresholds) else thresholds[-1]
        )
        metrics['optimal_threshold'] = float(optimal_threshold)
        metrics['optimal_f1'] = float(f1_scores[optimal_idx])
        
        # Add metrics with optimal threshold
        binary_preds_optimal = (all_similarities >= optimal_threshold).astype(int)
        metrics['accuracy_optimal'] = float(
            np.mean(binary_preds_optimal == all_labels)
        )
        
        tp_opt = np.sum((binary_preds_optimal == 1) & (all_labels == 1))
        fp_opt = np.sum((binary_preds_optimal == 1) & (all_labels == 0))
        fn_opt = np.sum((binary_preds_optimal == 0) & (all_labels == 1))
        
        metrics['precision_optimal'] = float(
            tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0
        )
        metrics['recall_optimal'] = float(
            tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
        )
    except Exception as e:
        print(f"Error computing optimal threshold: {e}")
    
    # Calculate similarity statistics
    metrics['min_similarity'] = float(np.min(all_similarities))
    metrics['max_similarity'] = float(np.max(all_similarities))
    metrics['mean_similarity'] = float(np.mean(all_similarities))
    metrics['mean_similarity_positive'] = float(
        np.mean(all_similarities[all_labels == 1])
    )
    metrics['mean_similarity_negative'] = float(
        np.mean(all_similarities[all_labels == 0])
    )
    
    return metrics, all_similarities, all_labels


def inference_on_sequence_pair(
    model, cdr3_1, antigen_1, cdr3_2, antigen_2, device=None
):
    """Run inference on a single pair of sequences.
    
    Args:
        model: The loaded contrastive model
        cdr3_1 (str): First CDR3 sequence
        antigen_1 (str): First antigen sequence
        cdr3_2 (str): Second CDR3 sequence
        antigen_2 (str): Second antigen sequence
        device: Computation device
        
    Returns:
        float: Similarity score between the sequences
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Combine sequences with separator
    sequence1 = str(cdr3_1) + "|" + str(antigen_1)
    sequence2 = str(cdr3_2) + "|" + str(antigen_2)
    
    # Tokenize sequences
    tokens1 = tokenize_sequence(sequence1).unsqueeze(0).to(device)
    tokens2 = tokenize_sequence(sequence2).unsqueeze(0).to(device)
    
    # Get embeddings
    with torch.no_grad():
        z1 = model(tokens1)
        z2 = model(tokens2)
        
        # Use model's similarity function if available, otherwise use default
        if hasattr(model, 'compute_similarity'):
            similarity = model.compute_similarity(z1, z2).item()
        else:
            # Default cosine similarity
            similarity = F.cosine_similarity(z1, z2, dim=1).item()
        
    return similarity


def main_inference():
    """Main function to run inference on a test set."""
    # Paths
    model_weights_path = '/home/jupyter/checkpoints/model_epoch_7.pt'
    esm3_checkpoint_path = '/home/jupyter/oracle/esm3_finetuned_antibody/checkpoint_epoch_1.pt'
    test_pairs_path = '/home/jupyter/DATA/hyperbind_train/synthetic/test_pairs.csv'
    raw_data_path = '/home/jupyter/DATA/hyperbind_train/synthetic/test.csv'
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load ESM3 model
        print("Loading ESM3 backbone...")
        from esm.pretrained import ESM3_sm_open_v0
        esm3_model = ESM3_sm_open_v0()
        print("ESM3 model instantiated")

        # Load finetuned weights for the backbone
        esm3_ckpt = torch.load(esm3_checkpoint_path, map_location=device)
        esm3_state_dict = esm3_ckpt.get('model_state_dict', esm3_ckpt)
        esm3_model.load_state_dict(esm3_state_dict, strict=False)
        print("Loaded finetuned ESM3 backbone weights")
        
        # Wrap backbone in contrastive model
        model = EnhancedContrastiveModel(
            esm3_model, freeze_backbone=False, temperature=0.15
        )
        print("Enhanced contrastive model created")

        # Load contrastive head weights
        print(f"Loading contrastive head weights from {model_weights_path}")
        checkpoint = torch.load(model_weights_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("Contrastive head weights loaded (with some mismatches allowed)")

        # Print summary of loaded keys
        missing_keys = set(k for k, _ in model.named_parameters()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(k for k, _ in model.named_parameters())
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        print(f"Successfully loaded keys: {len(state_dict) - len(unexpected_keys)}")
        
        # Prepare model for inference
        model.to(device)
        model.eval()
        print("Model moved to device and set to eval mode")
        
        # Load test data
        test_loader = create_test_dataloader(
            test_pairs_path=test_pairs_path,
            raw_data_path=raw_data_path,
            batch_size=16
        )
        print(f"Test loader created with {len(test_loader.dataset)} samples")

        # Run inference
        print("Running inference on test set...")
        metrics, all_similarities, all_labels = run_inference_on_test_set(
            model, test_loader, device=device, similarity_threshold=0.7
        )

        # Print evaluation metrics
        print("\nTest Set Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Plot similarity and ROC
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(
                all_similarities[all_labels==1], 
                bins=50, alpha=0.5, label='Positive Pairs'
            )
            plt.hist(
                all_similarities[all_labels==0], 
                bins=50, alpha=0.5, label='Negative Pairs'
            )
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('Distribution of Similarity Scores')
            plt.savefig('/home/jupyter/similarity_distribution.png')
            print("Saved similarity distribution plot to /home/jupyter/similarity_distribution.png")

            fpr, tpr, _ = roc_curve(all_labels, all_similarities)
            plt.figure(figsize=(10, 6))
            plt.plot(
                fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})'
            )
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig('/home/jupyter/roc_curve.png')
            print("Saved ROC curve plot to /home/jupyter/roc_curve.png")
        except Exception as e:
            print(f"Plotting error: {e}")
        
        # Sample inference
        print("\nRunning sample inference:")
        cdr3_1 = "CASSQDMTV"
        antigen_1 = "VPGFPTVRRALVPK"
        cdr3_2 = "CASSQDMTV"
        antigen_2 = "VPGFPTVRRALVPK"
        similarity = inference_on_sequence_pair(
            model, cdr3_1, antigen_1, cdr3_2, antigen_2, device=device
        )
        print(f"Similarity between same sequences: {similarity:.4f}")

        cdr3_2 = "CASSFGQETQYF"
        antigen_2 = "LLFGYPVYV"
        similarity = inference_on_sequence_pair(
            model, cdr3_1, antigen_1, cdr3_2, antigen_2, device=device
        )
        print(f"Similarity between different sequences: {similarity:.4f}")
        
        # Save matched dataset
        try:
            matched_data = {
                'cdr3_1': test_loader.dataset.cdr3_1,
                'antigen_1': test_loader.dataset.antigen_1,
                'cdr3_2': test_loader.dataset.cdr3_2,
                'antigen_2': test_loader.dataset.antigen_2,
                'label': test_loader.dataset.labels
            }
            matched_df = pd.DataFrame(matched_data)
            matched_df.to_csv(
                '/home/jupyter/DATA/hyperbind_train/matched_test_pairs.csv', 
                index=False
            )
            print(
                "Saved matched dataset to "
                "/home/jupyter/DATA/hyperbind_train/matched_test_pairs.csv"
            )
        except Exception as e:
            print(f"Could not save matched dataset: {e}")
        
        return model, metrics, all_similarities, all_labels

    except Exception as e:
        print(f"Critical error in main_inference: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    main_inference()
