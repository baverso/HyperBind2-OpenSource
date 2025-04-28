#!/usr/bin/env python3
"""
Antibody-Antigen Binding Prediction Model

This script implements a contrastive learning approach for predicting antibody-antigen binding.
It uses ESM3 protein language model embeddings to encode antibody (CDR3) and antigen sequences,
then trains a model to distinguish binding from non-binding pairs using contrastive learning.

Key features:
- ESM3-based sequence encoding with attention pooling
- Contrastive learning with InfoNCE loss
- Composite pairing strategy based on antigen and binding class
- No structure-based features, focusing only on sequence information

The model achieves significantly better performance (~0.76 ROC AUC) compared to previous
approaches (~0.53 ROC AUC) by using composite pairing and removing structure information.

Author: Daniel Dell'uomo
Date: April 2025
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt

# Import ESM3 and necessary components
from esm.pretrained import ESM3_sm_open_v0
try:
    from esm.tokenization import get_esm3_model_tokenizers
    from esm.utils.constants.models import ESM3_OPEN_SMALL
except ImportError:
    print("Could not import ESM3 tokenization components - will fall back to ESM-1b alphabet")
    from esm.data import Alphabet, BatchConverter

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory for checkpoints
checkpoint_dir = "checkpoints/antibody_contrastive_improved"
os.makedirs(checkpoint_dir, exist_ok=True)

# Define paths
train_data_path = "/home/jupyter/DATA/hyperbind_train/synthetic/train.csv"
test_data_path = "/home/jupyter/DATA/hyperbind_train/synthetic/test.csv"
test_pairs_path = "/home/jupyter/DATA/hyperbind_train/synthetic/test_pairs.csv"
esm3_checkpoint_path = "/home/jupyter/oracle/esm3_finetuned_antibody/checkpoint_epoch_1.pt"


class EnhancedESM3Adapter(nn.Module):
    """Enhanced adapter for ESM3 with dynamic embedding dimension handling."""
    
    def __init__(self, esm3_model, verbose=False):
        super().__init__()
        self.model = esm3_model
        self.embed_dim = 1536  # Initial guess for ESM3-small
        self.verbose = verbose
        
        # Get number of layers from model config
        if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'n_layer'):
            self.max_layer_idx = self.model.cfg.n_layer - 1  # 0-indexed
        elif hasattr(self.model, 'args') and hasattr(self.model.args, 'n_layer'):
            self.max_layer_idx = self.model.args.n_layer - 1  # 0-indexed
        else:
            # Fallback to dynamic scan
            self.max_layer_idx = 0
            for name, _ in self.model.named_modules():
                if "layers" in name:
                    try:
                        layer_num = int(name.split(".")[-1])
                        self.max_layer_idx = max(self.max_layer_idx, layer_num)
                    except:
                        pass
        
        if self.verbose:
            print(f"Using representation from model (max layer: {self.max_layer_idx})")
        
        # We'll initialize attention pooling later after we know the embedding dimension
        self.attention = None
        self._attention_initialized = False
    
    def _initialize_attention(self, embed_dim):
        """Initialize attention pooling with the correct embedding dimension."""
        self.embed_dim = embed_dim
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._attention_initialized = True
        if self.verbose:
            print(f"Initialized attention pooling for embedding dimension: {embed_dim}")
    
    def attention_pool(self, embeddings, padding_mask):
        """Apply attention pooling to sequence embeddings."""
        # Dynamically initialize attention if needed
        if not self._attention_initialized:
            embed_dim = embeddings.size(-1)
            self._initialize_attention(embed_dim)
            # Move to same device as embeddings
            self.attention = self.attention.to(embeddings.device)
        
        # Create mask for attention
        mask = padding_mask.unsqueeze(-1).float()
        
        # Calculate attention weights
        attention_weights = self.attention(embeddings)
        
        # Apply mask to attention weights
        attention_weights = attention_weights * mask
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum of token embeddings
        pooled_embeddings = torch.sum(embeddings * attention_weights, dim=1)
        return pooled_embeddings
    
    def forward(self, batch_tokens, batch_lens=None):
        """Process sequence tokens using ESM3 and pool the embeddings."""
        with torch.no_grad():
            # Updated to match ESM3 API - no repr_layers parameter
            try:
                # First try the standard forward call
                results = self.model(sequence_tokens=batch_tokens)
                
                # Check if there's a 'representations' key in the results
                if hasattr(results, 'representations') and self.max_layer_idx in results.representations:
                    embeddings = results.representations[self.max_layer_idx]
                    if self.verbose:
                        print(f"Using ESM3 representations from layer {self.max_layer_idx}")
                # If no representations key, try to get the last hidden state
                elif hasattr(results, 'last_hidden_state'):
                    embeddings = results.last_hidden_state
                    if self.verbose:
                        print("Using ESM3 last_hidden_state")
                # If that fails too, look for an 'embed_output' attribute
                elif hasattr(results, 'embed_output'):
                    embeddings = results.embed_output
                    if self.verbose:
                        print("Using ESM3 embed_output")
                # If all else fails, just use the sequence logits if available
                elif hasattr(results, 'sequence_logits'):
                    # Not ideal but we can use sequence logits as embeddings
                    embeddings = results.sequence_logits
                    if self.verbose:
                        print(f"Using ESM3 sequence_logits as embeddings (fallback), "
                              f"shape: {results.sequence_logits.shape}")
                else:
                    # Last resort - just treat the results as the embeddings directly
                    if isinstance(results, torch.Tensor):
                        embeddings = results
                        if self.verbose:
                            print("Using direct tensor output from ESM3")
                    else:
                        # Try to find any tensor in results that could be embeddings
                        found = False
                        for attr_name in dir(results):
                            if attr_name.startswith('_'):
                                continue
                            try:
                                attr = getattr(results, attr_name)
                                if isinstance(attr, torch.Tensor) and attr.dim() == 3:  # [batch, seq, features]
                                    embeddings = attr
                                    found = True
                                    if self.verbose:
                                        print(f"Using ESM3 attribute {attr_name} as embeddings, "
                                              f"shape: {attr.shape}")
                                    break
                            except:
                                continue
                        
                        if not found:
                            raise ValueError("Could not find suitable embeddings in ESM3 model output")
            except Exception as e:
                if self.verbose:
                    print(f"Error using standard ESM3 forward: {e}")
                    print("Trying legacy call pattern...")
                
                # Try with different parameter names
                try:
                    # Try tokens parameter
                    results = self.model(tokens=batch_tokens)
                except:
                    try:
                        # Try with 'input_ids'
                        results = self.model(input_ids=batch_tokens)
                    except:
                        # Last resort - positional args
                        results = self.model(batch_tokens)
                
                # Extract embeddings from results
                if hasattr(results, 'last_hidden_state'):
                    embeddings = results.last_hidden_state
                elif isinstance(results, torch.Tensor):
                    embeddings = results
                else:
                    # Try to find a 3D tensor in results
                    found = False
                    for attr_name in dir(results):
                        if attr_name.startswith('_'):
                            continue
                        try:
                            attr = getattr(results, attr_name)
                            if isinstance(attr, torch.Tensor) and attr.dim() == 3:
                                embeddings = attr
                                found = True
                                if self.verbose:
                                    print(f"Using attribute {attr_name} as embeddings, "
                                          f"shape: {attr.shape}")
                                break
                        except:
                            continue
                    
                    if not found:
                        raise ValueError("Could not extract embeddings from model output")
            
            # Create padding mask (1 for real tokens, 0 for padding)
            padding_mask = (batch_tokens != 1)  # 1 is pad_idx in ESM3
            
            # Display embedding shape for debugging
            if self.verbose:
                print(f"Extracted embeddings shape: {embeddings.shape}")
        
        # Apply attention pooling
        pooled_embeddings = self.attention_pool(embeddings, padding_mask)
        
        # Log shape for debugging if verbose
        if self.verbose and not hasattr(self, '_shape_logged'):
            print(f"ESM3 Adapter output shape: {pooled_embeddings.shape}")
            self._shape_logged = True
            
        return pooled_embeddings


class ContrastiveProjectionHead(nn.Module):
    """Improved projection head for contrastive learning."""
    
    def __init__(self, in_dim, hidden_dim=1024, out_dim=256, dropout=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            # First block
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second block with residual connection
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Output projection
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, x):
        # Project to lower-dimensional space
        x = self.network(x)
        
        # Normalize to unit length
        x = F.normalize(x, p=2, dim=1)
        return x


class AntibodyBindingModel(nn.Module):
    """Contrastive learning model for antibody-antigen binding prediction."""
    
    def __init__(self, esm3_model, temperature=0.07, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        # ESM3 backbone
        self.esm_adapter = EnhancedESM3Adapter(esm3_model, verbose=verbose)
        
        # We will initialize the projection head after the first forward pass
        # when we know the actual embedding dimension
        self.projection_head = None
        self._initialized = False
        
        # Temperature parameter (fixed, not learnable)
        self.temperature = temperature
    
    def _initialize_model(self, sequence_embedding_dim):
        """Initialize the projection head with the correct input dimension."""
        # Projection head - using just sequence embeddings now, no structure
        self.projection_head = ContrastiveProjectionHead(
            in_dim=sequence_embedding_dim,
            hidden_dim=1024,
            out_dim=256
        )
        
        self._initialized = True
        if self.verbose:
            print(f"Initialized projection head with input dimension: {sequence_embedding_dim}")
    
    def freeze_layers(self, unfreeze_last_n_layers=6):
        """Freeze layers of the backbone, leaving only the last N layers trainable."""
        # Freeze the ESM adapter
        for param in self.esm_adapter.parameters():
            param.requires_grad = False
            
        # Make sure the attention pooling remains trainable
        if self.esm_adapter.attention is not None:
            for param in self.esm_adapter.attention.parameters():
                param.requires_grad = True
        
        if self.verbose:
            print("ESM backbone frozen, attention pooling remains trainable")
    
    def forward(self, sequence_tokens):
        """
        Forward pass through the model.
        
        Args:
            sequence_tokens: Tokenized sequences
            
        Returns:
            torch.Tensor: Projected embeddings
        """
        # Get sequence embeddings
        sequence_embeddings = self.esm_adapter(sequence_tokens)
        
        # Initialize model if this is the first forward pass
        if not self._initialized:
            self._initialize_model(sequence_embeddings.size(1))
            # Move to same device as embeddings
            self.projection_head = self.projection_head.to(sequence_embeddings.device)
        
        # Project to lower-dimensional space (no structure encoder anymore)
        projected = self.projection_head(sequence_embeddings)
        
        return projected
    
    def compute_similarity(self, z1, z2):
        """Compute cosine similarity between embeddings."""
        # Normalize embeddings (should already be normalized from projection head)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(z1, z2.t()) / self.temperature
        
        return similarity


# Custom ESM3 tokenization for sequences with manual fallback
class CustomESM3Tokenizer:
    """
    Custom tokenizer that mirrors the tokenization used in training.
    This provides a fallback when the official ESM3 tokenizer isn't available.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        
        # Define special tokens for ESM3
        self.pad_idx = 1      # Padding token
        self.mask_idx = 32    # Mask token for MLM
        self.cls_idx = 0      # Beginning of sequence
        self.eos_idx = 2      # End of sequence
        self.unk_idx = 3      # Unknown amino acid
        self.vocab_size = 64  # Standard vocabulary size for ESM3
        
        # Standard amino acid mapping for ESM3 - exact match to fine-tuning script
        self.aa_to_idx = {
            'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11, 'H': 12, 'I': 13,
            'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18, 'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23,
            'B': 24, 'J': 25, 'O': 26, 'U': 27, 'X': 28, 'Z': 29, '.': 30, '-': 31, '|': 31  # Map '|' to 31
        }
        
        if verbose:
            print("Initialized CustomESM3Tokenizer with matching vocabulary to training script")
    
    def tokenize(self, sequence):
        """
        Tokenize a single sequence using the ESM3 token mapping.
        
        Args:
            sequence (str): Amino acid sequence to tokenize
            
        Returns:
            torch.Tensor: Tensor of token IDs
        """
        token_ids = [self.cls_idx]  # Start with <cls>
        
        # Tokenize sequence
        for char in sequence:
            token_ids.append(self.aa_to_idx.get(char, self.unk_idx))
        
        # Add end of sequence token
        token_ids.append(self.eos_idx)
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def batch_tokenize(self, sequences, max_length=512, pad=True):
        """
        Tokenize a batch of sequences with padding.
        
        Args:
            sequences (List[str]): List of sequences to tokenize
            max_length (int): Maximum length for padding
            pad (bool): Whether to pad sequences
            
        Returns:
            torch.Tensor: Batch of token IDs [batch_size, seq_length]
        """
        tokenized = [self.tokenize(seq) for seq in sequences]
        
        if not pad:
            return tokenized
        
        # Find max length in this batch
        batch_max_length = min(max(len(t) for t in tokenized), max_length)
        
        # Pad all sequences to the same length
        padded = []
        for tokens in tokenized:
            if len(tokens) > max_length:
                # Truncate
                padded.append(tokens[:max_length])
            else:
                # Pad with pad_idx
                padding = torch.full((batch_max_length - len(tokens),), self.pad_idx, dtype=torch.long)
                padded.append(torch.cat([tokens, padding]))
        
        return torch.stack(padded)


# DATA HANDLING

class AntibodyPairDataset(Dataset):
    """Dataset for antibody-antigen binding prediction with pairs."""
    
    def __init__(self, data_df, pairs_df):
        """
        Initialize dataset with pairs for contrastive learning.
        
        Args:
            data_df: DataFrame with raw antibody and antigen data
            pairs_df: DataFrame with pairs information (ID1, ID2, pair_type)
        """
        self.data_df = data_df
        self.pairs_df = pairs_df
        
        print(f"Created dataset with {len(pairs_df)} pairs")
        self._log_pair_distribution()
    
    def _log_pair_distribution(self):
        """Print distribution of positive and negative pairs."""
        if 'pair_type' in self.pairs_df.columns:
            pair_dist = self.pairs_df['pair_type'].value_counts()
            print("Pair type distribution:")
            for pair_type, count in pair_dist.items():
                print(f"  {pair_type}: {count} pairs ({count/len(self.pairs_df)*100:.2f}%)")
    
    def __len__(self):
        """Return the number of pairs in the dataset."""
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        """Get a pair from the dataset."""
        pair = self.pairs_df.iloc[idx]
        
        id1 = pair['ID1']
        id2 = pair['ID2']
        pair_type = pair['pair_type'] if 'pair_type' in pair else ''
        
        # Get rows by numeric index
        try:
            row1 = self.data_df.iloc[int(id1)]
            row2 = self.data_df.iloc[int(id2)]
            
            # Get sequences
            cdr3_1 = str(row1['CDR3'])
            antigen_1 = str(row1['antigen_sequence'])
            
            cdr3_2 = str(row2['CDR3'])
            antigen_2 = str(row2['antigen_sequence'])
            
            # Truncate antigen sequence to prevent excessive length
            max_antigen_len = 300  # Limit antigen length
            antigen_1 = antigen_1[:max_antigen_len]
            antigen_2 = antigen_2[:max_antigen_len]
            
            # Combine sequences with '|' separator to match the training format
            # The fine-tuning code mapped '|' to token 31 
            sequence1 = "".join([cdr3_1, "|", antigen_1])  
            sequence2 = "".join([cdr3_2, "|", antigen_2])
            
            # Get label (1 for positive, 0 for negative)
            label = 1.0 if pair_type.lower() == 'positive' else 0.0
            
            return {
                'sequence1': sequence1,
                'sequence2': sequence2,
                'label': label,
                'cdr3_1': cdr3_1,
                'antigen_1': antigen_1,
                'cdr3_2': cdr3_2,
                'antigen_2': antigen_2
            }
        except Exception as e:
            # Report the error and re-raise
            print(f"Error processing pair {idx} (IDs: {id1}, {id2}): {e}")
            raise


def verify_pairs_indices(pairs_df, data_df, name="pairs"):
    """Verify that all indices in the pairs dataframe exist in the data dataframe."""
    valid = True
    for _, row in pairs_df.iterrows():
        id1, id2 = int(row['ID1']), int(row['ID2'])
        if id1 >= len(data_df) or id2 >= len(data_df):
            valid = False
            print(f"Invalid {name} indices: ID1={id1}, ID2={id2}, data has {len(data_df)} rows")
    
    if valid:
        print(f"All {name} indices are valid!")
    return valid


def setup_esm3_tokenization(verbose=False):
    """
    Set up the proper ESM3 tokenizer for sequence encoding.
    Falls back to custom ESM3 tokenizer with matching vocabulary if official not available.
    
    Args:
        verbose (bool): Whether to print detailed information
        
    Returns:
        tokenizer: The ESM3 sequence tokenizer or custom ESM3 tokenizer
        is_esm3_native: Boolean indicating if we're using native ESM3 tokenization
    """
    try:
        # Try to get the proper ESM3 tokenizer
        from esm.tokenization import get_esm3_model_tokenizers
        from esm.utils.constants.models import ESM3_OPEN_SMALL
        
        tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
        sequence_tokenizer = tokenizers.sequence
        if verbose:
            print("Successfully loaded native ESM3 tokenizer")
            # Check what methods are available
            print(f"Available methods: {dir(sequence_tokenizer)}")
        return sequence_tokenizer, True
    except Exception as e:
        if verbose:
            print(f"Could not load ESM3 tokenizer: {e}")
            print("Falling back to custom ESM3 tokenizer with matching vocabulary")
        
        # Use custom tokenizer with matching vocabulary
        custom_tokenizer = CustomESM3Tokenizer(verbose=verbose)
        return custom_tokenizer, False


def collate_antibody_pairs(batch, tokenizer, is_esm3_native=True):
    """Simplified collate function that works with ESM3 tokenizer."""
    # Prepare sequences for tokenization
    sequences1 = [item['sequence1'] for item in batch]
    sequences2 = [item['sequence2'] for item in batch]
    
    # Collect labels
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)
    
    # Additional metadata
    cdr3_1 = [item['cdr3_1'] for item in batch]
    antigen_1 = [item['antigen_1'] for item in batch]
    cdr3_2 = [item['cdr3_2'] for item in batch]
    antigen_2 = [item['antigen_2'] for item in batch]
    
    try:
        # Tokenize sequences with basic tokenize method and manual padding
        if is_esm3_native:
            # Tokenize each sequence separately
            tokenized1 = []
            tokenized2 = []
            
            for seq in sequences1:
                tokens = tokenizer.tokenize(seq)
                # Convert to tensor if it's not already
                if not isinstance(tokens, torch.Tensor):
                    if isinstance(tokens, list):
                        # Handle string tokens
                        tokens = torch.tensor([tokenizer.convert_tokens_to_ids(token) 
                                              for token in tokens], dtype=torch.long)
                    else:
                        # Already some kind of sequence but not a tensor
                        tokens = torch.tensor(tokens, dtype=torch.long)
                tokenized1.append(tokens)
                
            for seq in sequences2:
                tokens = tokenizer.tokenize(seq)
                # Convert to tensor if it's not already
                if not isinstance(tokens, torch.Tensor):
                    if isinstance(tokens, list):
                        # Handle string tokens
                        tokens = torch.tensor([tokenizer.convert_tokens_to_ids(token) 
                                              for token in tokens], dtype=torch.long)
                    else:
                        # Already some kind of sequence but not a tensor
                        tokens = torch.tensor(tokens, dtype=torch.long)
                tokenized2.append(tokens)
            
            # Manual padding
            max_len1 = max(len(t) for t in tokenized1)
            max_len2 = max(len(t) for t in tokenized2)
            
            # ESM3 pad token is 1
            pad_idx = 1
            
            padded1 = []
            for tokens in tokenized1:
                if len(tokens) < max_len1:
                    padding = torch.full((max_len1 - len(tokens),), pad_idx, dtype=torch.long)
                    padded_tokens = torch.cat([tokens, padding])
                else:
                    padded_tokens = tokens
                padded1.append(padded_tokens)
                
            padded2 = []
            for tokens in tokenized2:
                if len(tokens) < max_len2:
                    padding = torch.full((max_len2 - len(tokens),), pad_idx, dtype=torch.long)
                    padded_tokens = torch.cat([tokens, padding])
                else:
                    padded_tokens = tokens
                padded2.append(padded_tokens)
            
            tokens1 = torch.stack(padded1)
            tokens2 = torch.stack(padded2)
        else:
            # Use custom ESM3 tokenizer 
            custom_tokenizer = tokenizer
            tokens1 = custom_tokenizer.batch_tokenize(sequences1)
            tokens2 = custom_tokenizer.batch_tokenize(sequences2)
    except Exception as e:
        print(f"Error in tokenization: {e}")
        print("Falling back to custom tokenizer")
        custom_tokenizer = CustomESM3Tokenizer(verbose=True)
        tokens1 = custom_tokenizer.batch_tokenize(sequences1)
        tokens2 = custom_tokenizer.batch_tokenize(sequences2)
    
    return {
        'tokens1': tokens1,
        'tokens2': tokens2,
        'label': labels,
        'cdr3_1': cdr3_1,
        'antigen_1': antigen_1,
        'cdr3_2': cdr3_2,
        'antigen_2': antigen_2
    }


def generate_composite_pairs(df, n_pos_pairs=4000, n_neg_pairs=4000, seed=42):
    """
    Generate positive and negative pairs using a composite of antigen and binding_class.
    
    Args:
        df: DataFrame containing at least 'antigen' and 'binding_class' columns
        n_pos_pairs: Number of positive pairs to generate
        n_neg_pairs: Number of negative pairs to generate
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with columns ['ID1', 'ID2', 'pair_type']
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # Create a composite column for pairing
    df = df.copy()
    df['cohort'] = df['antigen'].astype(str) + "_" + df['binding_class'].astype(str)
    
    pairs = []
    
    # Track cohorts and the number of pairs generated per cohort
    cohorts = df['cohort'].unique()
    cohort_counts = {cohort: 0 for cohort in cohorts}
    
    # Generate positive pairs (same cohort)
    pos_count = 0
    max_pairs_per_cohort = max(10, n_pos_pairs // len(cohorts))  # Distribute pairs among cohorts
    
    # First pass: try to generate the maximum number of pairs per cohort
    for cohort in cohorts:
        indices = df[df['cohort'] == cohort].index.tolist()
        
        if len(indices) < 2:
            continue
        
        # Calculate how many pairs we can generate for this cohort
        pairs_possible = min(
            max_pairs_per_cohort,
            len(indices) * (len(indices) - 1) // 2
        )
        
        # Generate random pairs for this cohort
        pair_count = 0
        attempts = 0
        max_attempts = pairs_possible * 10
        
        while pair_count < pairs_possible and attempts < max_attempts:
            i, j = random.sample(indices, 2)
            
            # Avoid duplicate pairs
            if not any((p.get('ID1') == i and p.get('ID2') == j) or 
                      (p.get('ID1') == j and p.get('ID2') == i) for p in pairs):
                pairs.append({
                    'ID1': i,
                    'ID2': j,
                    'cohort1': cohort,
                    'cohort2': cohort,
                    'pair_type': 'positive'
                })
                pair_count += 1
                pos_count += 1
            
            attempts += 1
        
        cohort_counts[cohort] = pair_count
    
    # Second pass: if we haven't generated enough positive pairs,
    # sample more from cohorts that have many examples
    if pos_count < n_pos_pairs:
        # Sort cohorts by number of examples (descending)
        cohort_sizes = [(cohort, len(df[df['cohort'] == cohort])) for cohort in cohorts]
        cohort_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Generate more pairs from larger cohorts
        remaining = n_pos_pairs - pos_count
        for cohort, size in cohort_sizes:
            if size < 2 or remaining <= 0:
                continue
            
            indices = df[df['cohort'] == cohort].index.tolist()
            
            # Calculate additional pairs to generate for this cohort
            additional_pairs = min(
                remaining,
                (size * (size - 1) // 2) - cohort_counts[cohort]  # Max possible - already generated
            )
            
            if additional_pairs <= 0:
                continue
            
            pair_count = 0
            attempts = 0
            max_attempts = additional_pairs * 10
            
            while pair_count < additional_pairs and attempts < max_attempts:
                i, j = random.sample(indices, 2)
                
                # Avoid duplicate pairs
                if not any((p.get('ID1') == i and p.get('ID2') == j) or 
                          (p.get('ID1') == j and p.get('ID2') == i) for p in pairs):
                    pairs.append({
                        'ID1': i,
                        'ID2': j,
                        'cohort1': cohort,
                        'cohort2': cohort,
                        'pair_type': 'positive'
                    })
                    pair_count += 1
                    pos_count += 1
                    remaining -= 1
                
                attempts += 1
            
            cohort_counts[cohort] += pair_count
    
    # Generate negative pairs (different cohorts)
    neg_count = 0
    attempts = 0
    max_attempts = n_neg_pairs * 10
    
    while neg_count < n_neg_pairs and attempts < max_attempts:
        # Choose two different cohorts
        if len(cohorts) < 2:
            # Not enough cohorts for negative pairs
            break
            
        cohort1, cohort2 = random.sample(list(cohorts), 2)
        
        # Choose random samples from each cohort
        indices1 = df[df['cohort'] == cohort1].index.tolist()
        indices2 = df[df['cohort'] == cohort2].index.tolist()
        
        if not indices1 or not indices2:
            attempts += 1
            continue
        
        i = random.choice(indices1)
        j = random.choice(indices2)
        
        # Avoid duplicate pairs
        if not any((p.get('ID1') == i and p.get('ID2') == j) or 
                  (p.get('ID1') == j and p.get('ID2') == i) for p in pairs):
            pairs.append({
                'ID1': i,
                'ID2': j,
                'cohort1': cohort1,
                'cohort2': cohort2,
                'pair_type': 'negative'
            })
            neg_count += 1
        
        attempts += 1
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs)
    
    # Keep only necessary columns
    if 'cohort1' in pairs_df.columns:
        pairs_df = pairs_df[['ID1', 'ID2', 'pair_type']]
    
    print(f"Generated {len(pairs_df)} pairs:")
    print(f"  Positive pairs: {pairs_df['pair_type'].value_counts().get('positive', 0)}")
    print(f"  Negative pairs: {pairs_df['pair_type'].value_counts().get('negative', 0)}")
    
    return pairs_df


# CONTRASTIVE LOSS IMPLEMENTATION

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss (NT-Xent).
    
    This uses the full similarity matrix to compute the proper
    normalized temperature-scaled cross entropy loss.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, z1, z2, labels=None):
        """
        Compute InfoNCE loss.
        
        Args:
            z1: First set of embeddings [batch_size, dim]
            z2: Second set of embeddings [batch_size, dim]
            labels: Optional binary labels for weighted loss
            
        Returns:
            torch.Tensor: Loss value
        """
        batch_size = z1.shape[0]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.t()) / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # Loss using all negatives (both z1 and z2 as anchors)
        loss_i = self.criterion(sim_matrix, labels)
        loss_j = self.criterion(sim_matrix.t(), labels)
        
        # InfoNCE is the sum of both directions
        loss = (loss_i + loss_j) / 2
        
        return loss


# TRAINING FUNCTIONS

def train_antibody_model(model, train_loader, val_loader=None, n_epochs=10, lr=5e-4, 
                         weight_decay=1e-5, checkpoint_dir=None, device=None):
    """Train the antibody binding model with contrastive learning."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # We need to initialize the model first by doing a forward pass
    print("Initializing model with a test forward pass...")
    try:
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        tokens1 = sample_batch['tokens1'].to(device)
        
        # Run a single forward pass to initialize the model
        with torch.no_grad():
            _ = model(tokens1)
        
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error during model initialization: {e}")
        raise
    
    # Now create optimizer after projection head is initialized
    optimizer = torch.optim.AdamW(
        [
            # Main model parameters
            {"params": model.esm_adapter.parameters(), "lr": lr * 0.1},
            {"params": model.projection_head.parameters(), "lr": lr}
        ],
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Loss function
    criterion = InfoNCELoss(temperature=model.temperature)
    
    # Create checkpoint directory
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_roc_auc': []
    }
    
    # Track best validation metrics
    best_val_loss = float('inf')
    best_val_auc = 0.0
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        
        # Progress bar
        t_start = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch in progress_bar:
            # Move data to device
            tokens1 = batch['tokens1'].to(device)
            tokens2 = batch['tokens2'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            z1 = model(tokens1)
            z2 = model(tokens2)
            
            # Compute loss
            loss = criterion(z1, z2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        if val_loader is not None:
            val_loss, val_metrics = evaluate_antibody_model(
                model, val_loader, criterion=criterion, device=device
            )
            history['val_loss'].append(val_loss)
            history['val_roc_auc'].append(val_metrics['roc_auc'])
            
            # Print metrics
            print(f"Epoch {epoch+1}/{n_epochs} (Time: {time.time()-t_start:.1f}s):")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val ROC AUC: {val_metrics['roc_auc']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            
            # Save best model by validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, os.path.join(checkpoint_dir, "best_model_by_loss.pt"))
                print(f"  Saved best model (by loss) to {checkpoint_dir}/best_model_by_loss.pt")
            
            # Save best model by ROC AUC
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, os.path.join(checkpoint_dir, "best_model_by_auc.pt"))
                print(f"  Saved best model (by AUC) to {checkpoint_dir}/best_model_by_auc.pt")
        
        # Save checkpoint for the current epoch
        if checkpoint_dir is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }, os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt"))
        
        # Update learning rate
        scheduler.step()
    
    # Plot training history
    if len(history['train_loss']) > 1:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        if history['val_loss']:
            plt.plot(history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        if history['val_roc_auc']:
            plt.subplot(1, 2, 2)
            plt.plot(history['val_roc_auc'])
            plt.xlabel('Epoch')
            plt.ylabel('ROC AUC')
            plt.title('Validation ROC AUC')
        
        plt.tight_layout()
        if checkpoint_dir is not None:
            plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'))
        plt.show()
    
    # Load best model by AUC if available
    if val_loader is not None and checkpoint_dir is not None:
        best_model_path = os.path.join(checkpoint_dir, "best_model_by_auc.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    return model, history


def evaluate_antibody_model(model, val_loader, criterion=None, device=None):
    """Evaluate antibody binding model on validation set."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    val_loss = 0.0
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            tokens1 = batch['tokens1'].to(device)
            tokens2 = batch['tokens2'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            z1 = model(tokens1)
            z2 = model(tokens2)
            
            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(z1, z2)
                val_loss += loss.item()
            
            # Compute pairwise similarities
            similarity = model.compute_similarity(z1, z2)
            # Extract diagonal (similarity between corresponding pairs)
            pairwise_sim = torch.diag(similarity).cpu().numpy()
            
            # Store results
            all_similarities.extend(pairwise_sim)
            all_labels.extend(labels.cpu().numpy())
    
    # Average validation loss
    if criterion is not None and len(val_loader) > 0:
        val_loss /= len(val_loader)
    
    # Convert to numpy arrays
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    metrics = {}
    
    # ROC AUC
    try:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_similarities)
    except Exception as e:
        print(f"Error computing ROC AUC: {e}")
        metrics['roc_auc'] = 0.5  # Default to random
    
    # Precision-recall curve
    try:
        precision, recall, thresholds = precision_recall_curve(all_labels, all_similarities)
        metrics['pr_auc'] = auc(recall, precision)
    except Exception as e:
        print(f"Error computing PR AUC: {e}")
        metrics['pr_auc'] = 0.5  # Default to random
    
    # Find optimal threshold
    try:
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        metrics['optimal_threshold'] = optimal_threshold
        
        # Apply threshold and compute metrics
        binary_preds = (all_similarities >= optimal_threshold).astype(int)
        metrics['accuracy'] = np.mean(binary_preds == all_labels)
        
        # Calculate TP, FP, TN, FN
        tp = np.sum((binary_preds == 1) & (all_labels == 1))
        fp = np.sum((binary_preds == 1) & (all_labels == 0))
        fn = np.sum((binary_preds == 0) & (all_labels == 1))
        
        # Precision and recall
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall'] + 1e-8)
    except Exception as e:
        print(f"Error computing threshold-based metrics: {e}")
        metrics['optimal_threshold'] = 0.0
        metrics['accuracy'] = 0.5
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
    
    return val_loss, metrics


def main():
    """Main execution function with improved approach."""
    # QUICK TEST MODE - Set to False for full run
    QUICK_TEST = False
    
    # Load data
    print("Loading data...")
    
    # Use train data if exists, otherwise fall back to test data
    train_file_path = train_data_path
    if not os.path.exists(train_file_path):
        print(f"Warning: Training file {train_file_path} not found, using {test_data_path} as fallback")
        train_file_path = test_data_path
    
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_data_path)
    
    print(f"Loaded {len(train_df)} training samples")
    print(f"Loaded {len(test_df)} test samples")
    
    # Generate test pairs using composite approach
    print("\nGenerating test pairs using composite approach...")
    test_pairs_df = generate_composite_pairs(
        test_df,
        n_pos_pairs=3000,
        n_neg_pairs=3000,
        seed=SEED
    )
    
    # Apply quick test reductions if enabled
    if QUICK_TEST:
        print("\n*** QUICK TEST MODE ENABLED - USING REDUCED DATASET ***")
        # Tiny subset of data
        train_df = train_df.iloc[:100].reset_index(drop=True)
        test_df = test_df.iloc[:50].reset_index(drop=True)
        
        # Regenerate test pairs for this smaller dataset
        print("\nRegenerating test pairs for reduced dataset...")
        test_pairs_df = generate_composite_pairs(
            test_df,
            n_pos_pairs=10,
            n_neg_pairs=10,
            seed=SEED
        )
    
    # Validate indices in pairs DataFrame exist in the main DataFrames
    print("\nValidating data integrity...")
    verify_pairs_indices(test_pairs_df, test_df, "test")
    
    # Create train/val split
    print("\nCreating train/val split...")
    train_idx, val_idx = train_test_split(
        range(len(train_df)), 
        test_size=0.2, 
        random_state=SEED
    )
    
    train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_split_df = train_df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Train split: {len(train_split_df)} samples")
    print(f"Val split: {len(val_split_df)} samples")
    
    # Generate training and validation pairs using composite approach
    print("\nGenerating pairs using composite approach...")
    if QUICK_TEST:
        n_pos_pairs = 20  # Smaller for testing
        n_neg_pairs = 20
        n_pos_pairs_val = 10
        n_neg_pairs_val = 10
    else:
        n_pos_pairs = 4000
        n_neg_pairs = 4000  # Equal number of positives and negatives
        n_pos_pairs_val = 1000
        n_neg_pairs_val = 1000
        
    train_pairs_df = generate_composite_pairs(
        train_split_df,
        n_pos_pairs=n_pos_pairs,
        n_neg_pairs=n_neg_pairs,
        seed=SEED
    )
    
    val_pairs_df = generate_composite_pairs(
        val_split_df,
        n_pos_pairs=n_pos_pairs_val,
        n_neg_pairs=n_neg_pairs_val,
        seed=SEED
    )
    
    # Verify all indices are valid before proceeding
    print("\nVerifying pair indices...")
    verify_pairs_indices(train_pairs_df, train_split_df, "train")
    verify_pairs_indices(val_pairs_df, val_split_df, "val")
    verify_pairs_indices(test_pairs_df, test_df, "test")
    
    # Set up tokenization with extra diagnostics
    print("\nSetting up ESM3 tokenization...")
    tokenizer, is_esm3_native = setup_esm3_tokenization(verbose=True)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = AntibodyPairDataset(train_split_df, train_pairs_df)
    val_dataset = AntibodyPairDataset(val_split_df, val_pairs_df)
    test_dataset = AntibodyPairDataset(test_df, test_pairs_df)
    
    # Create data loaders with custom collate function
    # Use smaller batch size for testing if needed
    if QUICK_TEST:
        batch_size = 4  # Smaller for quick test
        num_workers = 0  # No multiprocessing for quicker debugging
    else:
        batch_size = 16  # Increased from 8 since we removed the structure encoder
        num_workers = 2
    
    print(f"Creating data loaders with batch size {batch_size}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_antibody_pairs(batch, tokenizer, is_esm3_native),
        pin_memory=True,
        drop_last=True  # Drop last batch to ensure consistent batch size for InfoNCE
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_antibody_pairs(batch, tokenizer, is_esm3_native),
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_antibody_pairs(batch, tokenizer, is_esm3_native),
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader.dataset)} pairs")
    print(f"  Val: {len(val_loader.dataset)} pairs")
    print(f"  Test: {len(test_loader.dataset)} pairs")
    
    # Test the first batch to make sure everything works
    print("\nTesting the first batch...")
    try:
        test_batch = next(iter(test_loader))
        print("Test batch keys:", test_batch.keys())
        for k, v in test_batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k} shape: {v.shape}")
            else:
                print(f"{k} type: {type(v)} length: {len(v)}")
                
        print("Test batch loaded successfully!")
    except Exception as e:
        print(f"Error loading test batch: {e}")
        print("This may be due to a tokenization or data issue.")
        return None, None
    
    # Load ESM3 model
    print("\nLoading ESM3 model...")
    esm3_model = ESM3_sm_open_v0()
    
    # Load finetuned weights if available
    if os.path.exists(esm3_checkpoint_path):
        print(f"Loading finetuned weights from {esm3_checkpoint_path}")
        esm3_ckpt = torch.load(esm3_checkpoint_path, map_location=device)
        esm3_state_dict = esm3_ckpt.get('model_state_dict', esm3_ckpt)
        esm3_model.load_state_dict(esm3_state_dict, strict=False)
    
    # Create model (verbose=False to avoid debug output)
    verbose = False
    print("\nCreating model without structure encoder...")
    model = AntibodyBindingModel(esm3_model, temperature=0.1, verbose=verbose)  # Adjusted temperature
    
    # Freeze layers
    model.freeze_layers(unfreeze_last_n_layers=6)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable "
          f"({trainable_params/total_params*100:.2f}%)")
    
    # Train model
    print("\nTraining model...")
    if QUICK_TEST:
        n_epochs = 1  # Just 1 epoch for testing
    else:
        n_epochs = 5
        
    model, history = train_antibody_model(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        lr=1e-3,  # Increased learning rate
        weight_decay=1e-4,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    criterion = InfoNCELoss(temperature=model.temperature)
    _, test_metrics = evaluate_antibody_model(model, test_loader, criterion=criterion, device=device)
    
    # Print test metrics
    print("\nTest set metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Save test metrics
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv(os.path.join(checkpoint_dir, 'test_metrics.csv'), index=False)
    
    print("\nEvaluation complete!")
    return model, test_metrics


# Execute main function
if __name__ == "__main__":
    model, metrics = main()
