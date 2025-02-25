#!/usr/bin/env python3
"""
Module: antibody_structure_encoder.py

This module provides a singleton implementation of an antibody structure encoder
for fine-tuning applications. The Singleton Class Pattern is used to ensure that
only one instance of the encoder is created at the application layer (e.g., within
a Jupyter notebook). This approach prevents redundant instantiations, reduces memory
usage, and guarantees consistency in the encoding process across the entire application.

Usage:
    from antibody_structure_encoder import AntibodyStructureEncoder
    encoder = AntibodyStructureEncoder(device="cuda")
    encoded_tensor = encoder.encode(antibody)
"""

from esm.utils.encoding import tokenize_structure
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.pretrained import ESM3_structure_encoder_v0

class AntibodyStructureEncoder:
    """
    Singleton class for encoding antibody structures.

    This class encapsulates the ESM structure encoder along with its corresponding
    tokenizer. It ensures that only one instance is created per application run,
    which is especially beneficial in resource-constrained environments like Jupyter.

    Attributes:
        tokenizer (StructureTokenizer): The tokenizer for processing antibody structures.
        encoder (nn.Module): The ESM structure encoder.
        device (str): The device on which the encoder operates (e.g., "cuda" or "cpu").
    """
    _instance = None

    def __new__(cls, device="cuda"):
        if cls._instance is None:
            cls._instance = super(AntibodyStructureEncoder, cls).__new__(cls)
            cls._instance._initialize(device)
        return cls._instance

    def _initialize(self, device):
        """
        Initializes the singleton instance with the given device.

        Args:
            device (str): Device to run the encoder on ("cuda" or "cpu").
        """
        self.device = device
        self.tokenizer = StructureTokenizer()
        self.encoder = ESM3_structure_encoder_v0(device=device)
        self.encoder.train()  # Enable training mode for fine-tuning.

    def encode(self, antibody):
        """
        Encodes an ESMProtein (antibody) object into a tensor for fine-tuning.

        Args:
            antibody (ESMProtein): Antibody object with attributes `coordinates` and `sequence`.

        Returns:
            torch.Tensor: Encoded tensor representing antibody backbone coordinates.
        """
        encoded_coords, plddt, structure_tokens = tokenize_structure(
            antibody.coordinates,
            self.encoder,
            self.tokenizer,
            reference_sequence=antibody.sequence,
            add_special_tokens=True
        )
        return encoded_coords

if __name__ == "__main__":
    # Simple test to verify that the singleton pattern is working.
    print("Testing AntibodyStructureEncoder singleton instantiation...")
    encoder1 = AntibodyStructureEncoder(device="cuda")
    encoder2 = AntibodyStructureEncoder(device="cpu")  # The device parameter will be ignored on subsequent instantiation.
    print("Are both encoder instances the same object?", encoder1 is encoder2)