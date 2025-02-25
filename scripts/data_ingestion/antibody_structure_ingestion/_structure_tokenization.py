import gzip
import torch
import numpy as np
from Bio.PDB import MMCIFParser
from scipy.spatial.transform import Rotation


class PDB2Atom37:
    """
    Module for extracting structure-based tokens from protein structures.

    This module contains functions to compute local backbone frames,
    calculate relative orientations, and convert structure information
    into tokens for model input. It also includes an example of loading a
    structure from an MMCIF file and generating structure tokens.

    Dan, this is a rough working draft with plenty of inline comments and questions.
    Feel free to suggest improvements!

    """
    
    def convert_structure_to_atom37(structure, chain_id):
        """
        Convert a protein structure to an atom37 representation.

        The atom37 representation is a numpy array of shape (L, 37, 3),
        where L is the number of residues. Each residue is represented by 37 atoms;
        here we fill only the first three positions with N, CA, and C coordinates,
        and set the rest to a high value (np.inf) to indicate missing atoms.

        Parameters
        ----------
        structure : Bio.PDB.Structure.Structure
            The protein structure loaded via Bio.PDB.
        chain_id : str
            The chain identifier (e.g., 'A' or 'H') to extract.

        Returns
        -------
        np.ndarray
            An array of shape (L, 37, 3) representing the atom37 coordinates.
        """
        # Extract the first model and the specified chain
        model = next(structure.get_models())
        chain = model[chain_id]
        residues = list(chain.get_residues())
        L = len(residues)
        # Create an array of shape (L, 37, 3) filled with np.inf as a placeholder
        atom37 = np.full((L, 37, 3), np.inf, dtype=np.float32)
        # For demonstration, we fill only the first three positions with N, CA, and C coordinates.
        for i, residue in enumerate(residues):
            try:
                atom37[i, 0] = residue['N'].get_coord()   # N atom
                atom37[i, 1] = residue['CA'].get_coord()  # CA atom
                atom37[i, 2] = residue['C'].get_coord()   # C atom
            except KeyError:
                # If any of these atoms are missing, we leave that row unchanged.
                # Dan, should we log a warning here?
                continue
        return atom37



    def convert_structure_to_atom37_all(structure):
        """
        Convert a protein structure (with potentially multiple chains) to an atom37 representation.

        The atom37 representation is a numpy array of shape (L_total, 37, 3), where L_total is 
        the total number of residues across all chains plus extra rows for chain breaks.
        For each residue, we fill only the first three positions with the N, CA, and C coordinates,
        and set the remaining positions to np.inf to indicate missing atoms.
        Between chains, a special "chain break" row is inserted (a row of shape (37, 3) filled with np.inf).

        Parameters
        ----------
        structure : Bio.PDB.Structure.Structure
            The protein structure loaded via Bio.PDB (e.g., from an MMCIF file).

        Returns
        -------
        np.ndarray
            An array of shape (L_total, 37, 3) representing the concatenated atom37 coordinates,
            with chain breaks inserted between chains.

        Dan, let me know if this approach (using np.inf rows to mark chain breaks) works
        for our downstream processing.
        """
        # Extract the first model from the structure (typical in PDB/MMCIF files)
        model = next(structure.get_models())

        # Get all chains from the model
        chains = list(model.get_chains())

        # This list will hold atom37 arrays for each chain and chain break rows in between.
        atom37_list = []

        # Define a chain break row: here we use a row filled with np.inf.
        # This row will be inserted between chains.
        chain_break = np.full((1, 37, 3), np.inf, dtype=np.float32)

        # Iterate over each chain in the model
        for chain in chains:
            # Get all residues in the chain
            residues = list(chain.get_residues())
            L = len(residues)
            if L == 0:
                # Dan, should we warn if a chain is empty?
                continue

            # Create an atom37 array for the current chain: shape (L, 37, 3)
            # Initialize all positions with np.inf to indicate missing atoms.
            atom37_chain = np.full((L, 37, 3), np.inf, dtype=np.float32)

            # Loop over each residue in the chain
            for i, residue in enumerate(residues):
                try:
                    # Fill only the first three positions with N, CA, and C atom coordinates.
                    atom37_chain[i, 0] = residue['N'].get_coord()   # N atom coordinates
                    atom37_chain[i, 1] = residue['CA'].get_coord()  # CA atom coordinates
                    atom37_chain[i, 2] = residue['C'].get_coord()   # C atom coordinates
                except KeyError:
                    # If any required atom is missing, leave this residue's row unchanged.
                    # Dan, should we log a warning or count missing residues?
                    continue

            # Append the current chain's atom37 array to our list
            atom37_list.append(atom37_chain)
            # Insert a chain break row after each chain.
            atom37_list.append(chain_break)

        # Remove the last chain break if present
        if atom37_list and atom37_list[-1].shape == (1, 37, 3):
            atom37_list = atom37_list[:-1]

        # Concatenate all arrays along the first dimension to get the final atom37 representation.
        atom37_all = np.concatenate(atom37_list, axis=0)
        return atom37_all




    def get_backbone_frames(coords: torch.Tensor) -> torch.Tensor:
        """
        Compute local backbone frames from atomic coordinates.

        Each frame is computed using the positions of the N, CA, and C atoms.
        The frame axes are defined as:
          - t: normalized vector from CA to C (tangent)
          - n: normalized cross product of (N->CA) and (CA->C)
          - b: cross product of t and n (binormal)

        Parameters
        ----------
        coords : torch.Tensor
            Tensor of shape (n_residues, 3, 3) containing atomic coordinates.
            The second dimension corresponds to atoms in the order: N, CA, C.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_residues, 3, 3) where each 3x3 matrix is a
            rotation matrix representing the local backbone frame.
        """
        # Calculate vectors: N -> CA and CA -> C
        n_ca = coords[:, 1] - coords[:, 0]  # N -> CA vectors
        ca_c = coords[:, 2] - coords[:, 1]    # CA -> C vectors

        # Normalize vectors
        n_ca = n_ca / torch.norm(n_ca, dim=-1, keepdim=True)
        ca_c = ca_c / torch.norm(ca_c, dim=-1, keepdim=True)

        # Define frame axes
        t = ca_c
        n = torch.cross(n_ca, ca_c)
        n = n / torch.norm(n, dim=-1, keepdim=True)
        b = torch.cross(t, n)

        # Stack axes into rotation matrices (n, b, t as columns)
        frames = torch.stack([n, b, t], dim=-1)
        return frames


    def get_relative_orientations(frames: torch.Tensor) -> torch.Tensor:
        """
        Compute relative orientations (as quaternions) between backbone frames.

        For each pair of frames, the relative rotation matrix is computed,
        and then converted into a quaternion.

        Parameters
        ----------
        frames : torch.Tensor
            Tensor of shape (n_residues, 3, 3) containing rotation matrices.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_residues, n_residues, 4) containing quaternions
            that represent the relative rotations between backbone frames.
        """
        n_frames = frames.shape[0]

        # Calculate relative rotation matrices between all pairs
        # frames[i] * frames[j]^T gives the relative rotation from j to i.
        rel_rots = torch.matmul(
            frames.unsqueeze(1), frames.transpose(-2, -1).unsqueeze(0)
        )

        # Convert rotation matrices to quaternions
        rel_quats = torch.zeros((n_frames, n_frames, 4), device=frames.device)
        for i in range(n_frames):
            for j in range(n_frames):
                # Use CPU for conversion then move tensor to device
                r = Rotation.from_matrix(rel_rots[i, j].cpu().numpy())
                rel_quats[i, j] = torch.tensor(r.as_quat(), device=frames.device)

        return rel_quats


    def structure_to_tokens(
        coords: torch.Tensor, max_radius: float = 20.0, n_bins: int = 50
    ) -> torch.Tensor:
        """
        Convert protein structure into tokens for model input.

        This function computes CA-CA distance bins and discretized quaternion
        tokens from backbone frames, then combines them into a single token
        representation.

        Parameters
        ----------
        coords : torch.Tensor
            Tensor of shape (n_residues, 3, 3) containing atomic coordinates.
        max_radius : float, optional
            Maximum distance for binning CA-CA distances (default is 20.0).
        n_bins : int, optional
            Number of bins to discretize the distances (default is 50).

        Returns
        -------
        torch.Tensor
            Tensor containing combined structure tokens.
        """
        # Get CA coordinates (assuming CA is the second atom in the coordinate list)
        ca_coords = coords[:, 1]  # CA atoms

        # Compute pairwise distances between CA atoms
        dists = torch.cdist(ca_coords, ca_coords)

        # Compute backbone frames and relative orientations
        frames = get_backbone_frames(coords)
        orientations = get_relative_orientations(frames)

        # Discretize distances
        dist_bins = torch.linspace(0, max_radius, n_bins, device=coords.device)
        dist_tokens = torch.bucketize(dists.flatten(), dist_bins).reshape(dists.shape)

        # Discretize orientations using quaternion bins
        quat_bins = 8  # Number of bins per quaternion component
        # Adjust to ensure values in [0, quat_bins - 1]
        eps = 1e-6
        quat_tokens = torch.floor((orientations + 1 - eps) * quat_bins / 2).long()
        quat_tokens = (
            quat_tokens[..., 0] * quat_bins ** 3
            + quat_tokens[..., 1] * quat_bins ** 2
            + quat_tokens[..., 2] * quat_bins
            + quat_tokens[..., 3]
        )

        # Combine distance and orientation tokens into a single token
        structure_tokens = dist_tokens * quat_bins ** 4 + quat_tokens

        return structure_tokens


if __name__ == "__main__":
    # Define device for torch computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and process structure from a compressed MMCIF file
    cif_file_path = "/home/jupyter/dan/data/1bey.cif.gz"
    parser = MMCIFParser()
    with gzip.open(cif_file_path, "rt") as f:
        structure = parser.get_structure("1BEY", f)

    # Extract backbone coordinates
    # Each backbone is a list of [N, CA, C] coordinates per residue
    coords_list = []
    for model in structure:
        for chain in model:
            backbone = []
            for residue in chain:
                # Check if all required atoms are present
                if all(atom in residue for atom in ["N", "CA", "C"]):
                    backbone.append(
                        [
                            residue["N"].get_coord(),
                            residue["CA"].get_coord(),
                            residue["C"].get_coord(),
                        ]
                    )
            if backbone:
                coords_list.append(backbone)

    if not coords_list:
        raise ValueError("No backbone coordinates found in the structure.")

    # Convert coordinates to a torch tensor and move to the device
    coords_tensor = torch.tensor(
        coords_list[0], dtype=torch.float32
    ).to(device)

    # Generate structure tokens from the coordinates
    structure_tokens = structure_to_tokens(coords_tensor)
    print(f"Structure tokens shape: {structure_tokens.shape}")