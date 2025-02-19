# esm_worker.py
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=False
)

def embed_sequence(model, sequence: str):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    return output

def worker(gpu_id, model_name, sequences, return_dict):
    # Set the proper GPU for this process
    torch.cuda.set_device(gpu_id)
    # Each worker loads its own copy of the model on the correct GPU.
    model = ESM3.from_pretrained(model_name, device=torch.device(f"cuda:{gpu_id}"))
    results = [embed_sequence(model, seq) for seq in sequences]
    return_dict[gpu_id] = results