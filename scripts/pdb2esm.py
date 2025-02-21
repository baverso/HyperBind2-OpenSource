#!/usr/bin/env python3
"""
pdb2esm.py

This script parses PDB files and converts them into ESMProtein objects.
Depending on the structure, the PDB file is interpreted either as a monomer
(single chain) or a multimer (protein complex).

It uses the esm.sdk.api and esm.utils modules for converting PDB data into
protein objects, along with Biopython's PDBParser for reading the file.

Usage:
    python parse_pdb.py /path/to/pdbfile.pdb

Dependencies:
    - esm.sdk.api
    - esm.utils.structure.protein_chain
    - esm.utils.types
    - Bio (Biopython)
"""

import os
import sys
import argparse
from esm.sdk.api import ESMProtein, ProteinComplex
from esm.utils.structure.protein_chain import ProteinChain
# Note: FunctionAnnotation is imported from esm.utils.types for potential future use.
from esm.utils.types import FunctionAnnotation
from Bio import PDB


def read_multimer_structure(pdb_input):
    """
    Reads and processes a multimer PDB structure.

    This function reads a PDB file representing a protein complex (multimer)
    and converts it into an ESMProtein object by first creating a ProteinComplex
    object.

    Args:
        pdb_input (str): Path to the PDB file.

    Returns:
        ESMProtein: Processed protein structure, or None if processing fails.
    """
    try:
        protein_complex = ProteinComplex.from_pdb(pdb_input)
        multimer_protein = ESMProtein.from_protein_complex(protein_complex)
        return multimer_protein
    except Exception as e:
        print(f"❌ Error processing multimer structure: {e}")
        return None


def read_monomer_structure(pdb_input, chain):
    """
    Reads and processes a monomer PDB structure for a given chain.

    This function extracts a specific chain from a PDB file and converts it into
    an ESMProtein object by creating a ProteinChain object.

    Args:
        pdb_input (str): Path to the PDB file.
        chain (str): Chain ID to process.

    Returns:
        ESMProtein: Processed monomer structure, or None if processing fails.
    """
    try:
        protein_chain = ProteinChain.from_pdb(pdb_input, chain_id=chain)
        monomer_protein = ESMProtein.from_protein_chain(protein_chain)
        return monomer_protein
    except Exception as e:
        print(f"❌ Error processing monomer structure (Chain {chain}): {e}")
        return None


def detect_and_process_structure(pdb_input):
    """
    Detects the type of structure in a PDB file and processes it accordingly.

    This function uses Biopython's PDBParser to read the structure and determine
    if it contains multiple chains (multimer) or a single chain (monomer). It then
    calls the appropriate function to process the PDB file.

    Args:
        pdb_input (str): Path to the PDB file.

    Returns:
        ESMProtein: Processed protein structure, or None if processing fails.
    """
    parser = PDB.PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("protein", pdb_input)
        model = structure[0]  # Assume single-model PDB file

        # Extract all available chain IDs
        chains = [chain.id.strip() for chain in model.get_chains()]
        num_chains = len(chains)

        if num_chains == 0:
            print(f"❌ Error: No valid protein chains found in {pdb_input}.")
            return None

        if num_chains > 1:
            print(f"✅ Detected {num_chains} chains ({chains}). "
                  "Processing as a multimer.")
            return read_multimer_structure(pdb_input)
        elif num_chains == 1:
            chain_id = chains[0]
            if chain_id:
                print(f"✅ Detected a single chain ({chain_id}). "
                      "Processing as a monomer.")
                return read_monomer_structure(pdb_input, chain_id)
            else:
                print(f"⚠️ Error: Chain ID not properly detected in {pdb_input}.")
                return None

    except FileNotFoundError:
        print(f"❌ Error: PDB file not found: {pdb_input}")
    except Exception as e:
        print(f"❌ Unexpected error reading PDB file {pdb_input}: {e}")

    return None


def main():
    """
    Main function that parses command-line arguments and processes the PDB file.
    """
    arg_parser = argparse.ArgumentParser(
        description="Parse a PDB file into an ESMProtein object."
    )
    arg_parser.add_argument(
        "pdb_file",
        type=str,
        help="Path to the PDB file to be processed."
    )
    args = arg_parser.parse_args()

    pdb_file = args.pdb_file
    if not os.path.isfile(pdb_file):
        print(f"❌ Error: The file {pdb_file} does not exist.")
        sys.exit(1)

    protein = detect_and_process_structure(pdb_file)
    if protein is None:
        print("❌ Protein processing failed.")
    else:
        print("✅ Protein processing successful.")
        # Additional processing or output can be added here.


if __name__ == "__main__":
    main()