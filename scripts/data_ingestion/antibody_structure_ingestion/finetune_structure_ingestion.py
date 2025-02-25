#!/usr/bin/env python3
"""
Module for encoding and ingesting protein structure data for fine-tuning.
This file includes:
  - encode_protein: Converts an ESMProtein object into an encoded tensor.
  - ProteinStructureDataset: A PyTorch Dataset for processing PDB files.
  - create_dataloaders: Convenience function to build training and validation DataLoaders.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

# Import necessary functions and classes from the codebase
from esm.sdk.api import ESMProtein
from esm.utils.encoding import tokenize_structure  # Converts raw structure to tokens/tensors
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.pretrained import ESM3_structure_encoder_v0

# Insert the path for the antibody structure ingestion scripts
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'scripts', 'data_ingestion', 'antibody_structure_ingestion')))
from pdb2esm import read_monomer_structure, read_multimer_structure, detect_and_process_structure

# Global Variables
TRAIN_DIRECTORY = '/home/jupyter/DATA/hyperbind_train/sabdab/all_structures/train-test-split/'

# Initialize the structure tokenizer and encoder
structure_tokenizer = StructureTokenizer()
structure_encoder = ESM3_structure_encoder_v0(device="cuda")  # or "cpu"
structure_encoder.train()  # Set to training mode for fine-tuning

def encode_protein(protein: ESMProtein):
    """
    Encodes an ESMProtein object into tensors suitable for structure fine-tuning.
    
    Args:
        protein (ESMProtein): Protein object with attributes `coordinates` and `sequence`.
    
    Returns:
        torch.Tensor: Encoded coordinates tensor.
    """
    encoded_coords, plddt, structure_tokens = tokenize_structure(
        protein.coordinates,
        structure_encoder,
        structure_tokenizer,
        reference_sequence=protein.sequence,
        add_special_tokens=True
    )
    return encoded_coords

class ProteinStructureDataset(Dataset):
    """
    Custom Dataset for loading and encoding protein structures.
    
    Each item is a tuple of (encoded_protein, ground_truth_coordinates).
    """
    def __init__(self, pdb_directory: str, suffix: str, encoder):
        """
        Args:
            pdb_directory (str): Directory containing PDB files.
            suffix (str): File suffix to filter PDB files (e.g. "_train.pdb" or "_val.pdb").
            encoder (callable): Function to encode an ESMProtein into a tensor.
        """
        self.pdb_directory = pdb_directory
        self.pdb_files = [
            os.path.join(pdb_directory, f)
            for f in os.listdir(pdb_directory) if f.endswith(suffix)
        ]
        self.encoder = encoder

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        pdb_path = self.pdb_files[idx]
        # Process the PDB file into an ESMProtein object.
        protein = detect_and_process_structure(pdb_path) # our custom pdb parser, detects Vh/Vl multimers
        if protein is None:
            raise ValueError(f"Protein processing failed for {pdb_path}")
        # Ground truth coordinates from the processed protein.
        gt_coords = protein.coordinates
        # Encode the protein using the provided encoder.
        encoded_protein = self.encoder(protein)
        return encoded_protein, gt_coords

def create_dataloaders(pdb_directory: str, batch_size: int = 2):
    """
    Creates DataLoaders for training and validation datasets.
    
    Args:
        pdb_directory (str): Directory containing PDB files.
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = ProteinStructureDataset(pdb_directory, "_train.pdb", encoder=encode_protein)
    val_dataset = ProteinStructureDataset(pdb_directory, "_val.pdb", encoder=encode_protein)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Optional: test the ingestion functions when running this module directly.
if __name__ == '__main__':
    train_loader, val_loader = create_dataloaders(TRAIN_DIRECTORY, batch_size=2)
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")