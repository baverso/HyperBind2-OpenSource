#!/usr/bin/env python
"""
assemble_atom37_multimer.py

This script reads a PDB file and assembles an atom37 multimer tensor by 
extracting the atom37 representation for each chain specified by the user,
inserting a chain break row between chains, and concatenating the results.
The final output is a torch tensor of shape (total_residues, 37, 3).

Usage:
    python assemble_atom37_multimer.py --pdb_path /path/to/file.pdb --chains H L
"""

import argparse
import logging
import numpy as np
import torch
from esm.utils.structure.protein_chain import ProteinChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def assemble_atom37_multimer(pdb_path: str, chain_ids: list) -> torch.Tensor:
    """
    Assemble an atom37 multimer tensor from a PDB file for specified chains.
    
    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    chain_ids : list of str
        List of chain identifiers to include (e.g., ['H', 'L']).
        
    Returns
    -------
    torch.Tensor
        A tensor of shape (total_residues, 37, 3) representing the multimer.
        A chain-break row (filled with np.inf) is inserted between chains.
    """
    multimer_list = []
    # Define a chain-break row. This row (1, 37, 3) filled with np.inf will separate chains.
    chain_break = np.full((1, 37, 3), np.inf, dtype=np.float32)
    
    for i, chain_id in enumerate(chain_ids):
        logger.info(f"Processing chain '{chain_id}' from {pdb_path}...")
        try:
            # Create a ProteinChain object for the given chain
            protein_chain = ProteinChain.from_pdb(path=pdb_path, chain_id=chain_id)
        except Exception as e:
            logger.error(f"Error processing chain '{chain_id}': {e}")
            raise
        
        # Ensure that an atom37 representation was successfully created
        if protein_chain.atom37 is None:
            logger.error(f"No atom37 representation found for chain '{chain_id}'.")
            raise ValueError(f"No atom37 representation found for chain '{chain_id}'.")
        
        # Append the atom37 representation for this chain
        multimer_list.append(protein_chain.atom37)
        # Insert a chain-break row if this is not the last chain
        if i < len(chain_ids) - 1:
            multimer_list.append(chain_break)
    
    # Concatenate the representations along the residue (first) dimension
    atom37_multimer = np.concatenate(multimer_list, axis=0)
    # Convert the result to a torch tensor
    atom37_multimer_tensor = torch.tensor(atom37_multimer, dtype=torch.float32)
    
    return atom37_multimer_tensor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble an atom37 multimer tensor from a PDB file for specified chains."
    )
    parser.add_argument(
        "--pdb_path", type=str, required=True,
        help="Path to the PDB file."
    )
    parser.add_argument(
        "--chains", type=str, nargs="+", required=True,
        help="List of chain identifiers to include (e.g., H L)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # Assemble the atom37 multimer tensor using the provided PDB and chain list.
    atom37_multimer_tensor = assemble_atom37_multimer(args.pdb_path, args.chains)
    logger.info(f"Atom37 multimer tensor shape: {atom37_multimer_tensor.shape}")
    # Here you can save or pass the tensor to your model.
    # For example, to save:
    # torch.save(atom37_multimer_tensor, "atom37_multimer.pt")
    print("Assembled atom37 multimer tensor:")
    print(atom37_multimer_tensor)

if __name__ == "__main__":
    main()