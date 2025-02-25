#!/usr/bin/env python3
"""
Module for loading the pre-trained ESM3 model for fine-tuning.
This file includes the create_esm3_model function.
"""

import torch
from esm.pretrained import ESM3_sm_open_v0

def create_esm3_model(device: torch.device):
    """
    Loads and prepares the ESM3 model for fine-tuning.
    
    Args:
        device (torch.device): Device on which to load the model.
    
    Returns:
        torch.nn.Module: The loaded ESM3 model in training mode.
    """
    # Instantiate the model.
    model = ESM3_sm_open_v0()
    
    # Load the pre-trained weights.
    state_dict = torch.load(
        "/home/jupyter/DATA/model_weights/esm3_complete/esm3_sm_open_v1_state_dict.pt",
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()  # Set the model to training mode.
    return model

# Optional: test the model loader when running this module directly.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_esm3_model(device)
    print("ESM3 model loaded successfully on", device)