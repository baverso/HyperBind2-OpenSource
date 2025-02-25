#!/usr/bin/env python3
"""
Compute global RMSD for aligned protein structure pairs.

This module reorders ground-truth and inferred Bio.PDB Structure objects by their
structure IDs, aligns each inferred structure to its corresponding ground truth using
global CA atom alignment (ignoring chain boundaries), and computes the global RMSD for
each alignment. Additionally, it determines whether the structure is a monomer or a multimer
(based on the number of chains in the ground truth).

Example usage in a Jupyter Notebook:
    from rmsd_table import compute_alignment_rmsd_table
    results = compute_alignment_rmsd_table(ground_truth_structures, inferred_list)
    import pandas as pd
    df = pd.DataFrame(results)
    display(df)
"""

from io import StringIO
from Bio.PDB import PDBIO, Superimposer


def reorder_structures(ground_truth_structures, inferred_list):
    """
    Reorder structure lists based on common structure IDs.

    Args:
        ground_truth_structures (list): List of Bio.PDB Structure objects (ground truth).
        inferred_list (list): List of Bio.PDB Structure objects (inferred).

    Returns:
        tuple: Two lists (ordered_ground_truth, ordered_inferred) containing the structures
               sorted by their common structure IDs.
    """
    gt_dict = {structure.get_id(): structure for structure in ground_truth_structures}
    inf_dict = {structure.get_id(): structure for structure in inferred_list}
    common_ids = sorted(set(gt_dict.keys()) & set(inf_dict.keys()))
    ordered_ground_truth = [gt_dict[sid] for sid in common_ids]
    ordered_inferred = [inf_dict[sid] for sid in common_ids]
    return ordered_ground_truth, ordered_inferred


def extract_all_ca_atoms(structure):
    """
    Extract all CA atoms from all chains in the first model of a structure.

    Args:
        structure (Bio.PDB.Structure): A Bio.PDB Structure object.

    Returns:
        list: A list of CA atoms encountered.
    """
    ca_atoms = []
    model = structure[0]  # assume single-model structure
    for chain in model:
        for residue in chain:
            if residue.has_id("CA"):
                ca_atoms.append(residue["CA"])
    return ca_atoms


def structure_to_pdb_string(structure):
    """
    Convert a Bio.PDB Structure object to a PDB-formatted string.

    Args:
        structure (Bio.PDB.Structure): A Bio.PDB Structure object.

    Returns:
        str: A string containing the PDB representation.
    """
    sio = StringIO()
    io = PDBIO()
    io.set_structure(structure)
    io.save(sio)
    pdb_str = sio.getvalue()
    sio.close()
    return pdb_str


def align_structures(gt_structure, inf_structure):
    """
    Align the inferred structure onto the ground truth structure using CA atoms.

    The alignment uses all CA atoms from all chains in the first model,
    ignoring chain boundaries. If the number of CA atoms differ, the first n atoms
    (n = minimum count) are used.

    Args:
        gt_structure (Bio.PDB.Structure): Ground truth structure.
        inf_structure (Bio.PDB.Structure): Inferred structure (modified in place).

    Returns:
        float: The global RMSD value of the alignment.

    Raises:
        ValueError: If one of the structures has no CA atoms.
    """
    gt_atoms = extract_all_ca_atoms(gt_structure)
    inf_atoms = extract_all_ca_atoms(inf_structure)

    if not gt_atoms or not inf_atoms:
        raise ValueError("One of the structures has no CA atoms.")

    if len(gt_atoms) != len(inf_atoms):
        min_count = min(len(gt_atoms), len(inf_atoms))
        print(f"Warning: CA atom counts differ ({len(gt_atoms)} vs {len(inf_atoms)}). "
              f"Using first {min_count} atoms.")
        gt_atoms = gt_atoms[:min_count]
        inf_atoms = inf_atoms[:min_count]

    sup = Superimposer()
    sup.set_atoms(gt_atoms, inf_atoms)
    sup.apply(inf_structure.get_atoms())
    return sup.rms


def compute_alignment_rmsd_table(ground_truth_structures, inferred_list):
    """
    Compute the global RMSD for each aligned structure pair and record if it is a monomer
    or a multimer (based on the number of chains in the ground truth).

    For each common structure ID, the inferred structure is aligned to the ground truth
    using CA atoms (ignoring chain boundaries), and the global RMSD is recorded.

    Args:
        ground_truth_structures (list): List of Bio.PDB Structure objects (ground truth).
        inferred_list (list): List of Bio.PDB Structure objects (inferred).

    Returns:
        list: A list of dictionaries with keys:
              "Structure ID", "Type", and "Global RMSD".
    """
    ordered_gt, ordered_inf = reorder_structures(ground_truth_structures, inferred_list)
    results = []
    for gt_structure, inf_structure in zip(ordered_gt, ordered_inf):
        gt_id = gt_structure.get_id()
        try:
            rmsd = align_structures(gt_structure, inf_structure)
            print(f"Aligned {gt_id}: Global RMSD = {rmsd:.2f} Å")
        except ValueError as ve:
            print(f"Skipping {gt_id}: {ve}")
            continue

        # Determine if the structure is a monomer or multimer based on the number of chains.
        chains = list(gt_structure[0].get_chains())
        structure_type = "Monomer" if len(chains) == 1 else "Multimer"
        results.append({"Structure ID": gt_id, "Type": structure_type, "Global RMSD": rmsd})
    return results


if __name__ == "__main__":
    # Example usage:
    # Replace the following with your lists of Bio.PDB Structure objects.
    ground_truth_structures = []  # e.g., loaded via Bio.PDB.PDBParser
    inferred_list = []            # e.g., loaded via Bio.PDB.PDBParser
    table = compute_alignment_rmsd_table(ground_truth_structures, inferred_list)
    for row in table:
        print(row)