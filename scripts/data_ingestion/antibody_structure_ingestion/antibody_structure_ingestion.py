#!/usr/bin/env python3
"""
Module for ingesting antibody structure data for fine-tuning.

This file includes:
  - AntibodyStructureDataset: A PyTorch Dataset for processing PDB files containing raw antibody data.
  - create_dataloaders: A convenience function to build training and validation DataLoaders.
  - custom_collate_fn: A custom collate function for raw antibody objects.

The purpose of this module is to pass raw or minimally processed antibody data directly 
into the deep learning model. The model is responsible for tokenizing and encoding the inputs 
during its forward pass, allowing end-to-end fine-tuning of both the encoder and transformer 
components.

Usage:
  python antibody_structure_ingestion.py --pdb_dir <PDB_DIRECTORY> [--batch_size 2]
"""

import os
import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

# Import necessary functions from the codebase.
from esm.sdk.api import ESMProtein

sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'scripts', 'data_ingestion', 'antibody_structure_ingestion')))
from pdb2esm import read_monomer_structure, read_multimer_structure, detect_and_process_structure

class AntibodyStructureDataset(Dataset):
    """
    Custom Dataset for loading raw antibody structure data.

    Each item in the dataset is a raw antibody object (e.g., an ESMProtein) produced by processing a PDB file.
    This antibody object contains attributes such as `coordinates` and `sequence` that will later be tokenized 
    and encoded by the deep learning model during training.
    """
    def __init__(self, pdb_directory: str, suffix: str):
        """
        Initializes the dataset by scanning for PDB files with a given suffix.

        Args:
            pdb_directory (str): Directory containing PDB files.
            suffix (str): File suffix to filter PDB files (e.g., "_train.pdb" or "_val.pdb").
        """
        self.pdb_directory = pdb_directory
        self.pdb_files = [
            os.path.join(pdb_directory, f)
            for f in os.listdir(pdb_directory) if f.endswith(suffix)
        ]

    def __len__(self):
        """Returns the number of PDB files found."""
        return len(self.pdb_files)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single PDB file to produce a raw antibody object.

        Args:
            idx (int): Index of the sample.

        Returns:
            antibody (ESMProtein): A raw antibody object containing attributes (e.g., `coordinates`, `sequence`)
                                   that will be further processed by the model.
        """
        pdb_path = self.pdb_files[idx]
        antibody = detect_and_process_structure(pdb_path)
        if antibody is None:
            raise ValueError(f"Antibody processing failed for {pdb_path}")
        return antibody

def custom_collate_fn(batch):
    """
    Custom collate function for raw antibody data.
    
    NOTE: This is just a placeholder function, and only useful with irregular tensors as a result of encoding.

    Since antibody objects may contain variable-length attributes (such as coordinate tensors),
    this function simply returns a list of antibody objects. Any necessary padding (e.g., for coordinate tensors)
    should be handled within the model during its forward pass.

    Args:
        batch (list): A list of raw antibody objects from the dataset.

    Returns:
        list: A list of raw antibody objects.
    """
    return batch

def create_dataloaders(pdb_directory: str, batch_size: int = 2):
    """
    Creates DataLoaders for training and validation datasets.

    Args:
        pdb_directory (str): Directory containing PDB files.
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: (train_loader, val_loader) for training and validation data.
    """
    train_dataset = AntibodyStructureDataset(pdb_directory, "_train.pdb")
    val_dataset = AntibodyStructureDataset(pdb_directory, "_val.pdb")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader, val_loader

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ingest raw antibody structure data for fine-tuning by processing PDB files."
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        required=True,
        help="Path to the directory containing PDB files for training and validation."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for DataLoaders (default: 2)."
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Create DataLoaders using the user-provided PDB directory.
    train_loader, val_loader = create_dataloaders(args.pdb_dir, batch_size=args.batch_size)
    
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")