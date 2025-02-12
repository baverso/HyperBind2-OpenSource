#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch
from torch.serialization import safe_globals

from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


# In[35]:


# Register the ESM3 class as safe for unpickling.
torch.serialization.add_safe_globals([ESM3])


# In[7]:


https://github.com/evolutionaryscale/esm/issues/178


# In[36]:


backbone_save_path = "/home/jupyter/DATA/evqlv-dev/model-weights/esm3_backbone/esm3_backbone_model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
with safe_globals([("esm.models.esm3.ESM3", ESM3)]):
    loaded_backbone_model = torch.load(backbone_save_path, map_location=device, weights_only=False)
    loaded_backbone_model.eval()


# In[37]:


state_dict = loaded_backbone_model.state_dict()
for key, tensor in state_dict.items():
    print(f"{key}: {tensor.shape}")
    
total_params = sum(p.numel() for p in loaded_backbone_model.parameters())
print("Total number of parameters:", total_params)


# # Validate encoder is working

# In[6]:


from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


# In[40]:


# for ESM3
# Initialize sequence 
seq_tokenizer = EsmSequenceTokenizer()
test_sequence = "CSSDGSYGFGAMDYW"
seq_tokens = seq_tokenizer.encode(test_sequence)
seq_tokens_tensor = torch.tensor(seq_tokens, dtype=torch.int64).unsqueeze(0).to(device)

# Create required tensors
dummy_average_plddt = torch.ones(seq_tokens_tensor.shape, dtype=torch.float32, device=device)
dummy_per_res_plddt = torch.ones(seq_tokens_tensor.shape, dtype=torch.float32, device=device)
dummy_structure_tokens = torch.zeros(seq_tokens_tensor.shape, dtype=torch.int64, device=device)

# Run forward pass
with torch.no_grad():
    embeddings = loaded_backbone_model.encoder.sequence_embed(seq_tokens_tensor)
    print("Embeddings shape:", embeddings.shape)
    
    
/////// validate other embedding types work (structure, etc. for ESM3)
    
    


# In[ ]:





# In[49]:


loaded_backbone_model.encoder.


# In[53]:


loaded_backbone_model.encoder.plddt_projection


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


embeddings


# In[ ]:





# In[ ]:


# objectives:
# 1. validate other embeddings work (may need antibody pdbs as input)
# 2. validate embedding dimensions / shape
# 3. Run multiple sequences in one pass?
# 4. Speed test 10, 100, 1000 sequences runtime
# 5. Training...OPT 3 guide a fine-tuning of this model (on 5 random antibody sequences)
# 6. LORA for training econo.y






# In[ ]:





# In[48]:


structure_embed = structure_tokens_embed(structure_tokens)
# ss8_embed = self.ss8_embed(ss8_tokens)
# sasa_embed = self.sasa_embed(sasa_tokens)
# plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt))
# sequence_embed = self.sequence_embed(sequence_tokens)


# In[38]:


# Original Code


seq_tokenizer = EsmSequenceTokenizer()

# Define a test protein sequence.
test_sequence = "CSSDGSYGFGAMDYW"

# Encode the sequence into token IDs.
seq_tokens = seq_tokenizer.encode(test_sequence)

# Convert tokens to a torch tensor with a batch dimension.

seq_tokens_dtype = torch.int64
seq_tokens_tensor = torch.tensor(seq_tokens, dtype=seq_tokens_dtype).unsqueeze(0).to(device)




### CREATING DUMMY TENSORS PER GPT o3 RECOMMENDARTION ###
# --- Create dummy tensors for required structure inputs ---
# Determine the dtype expected for plddt inputs from the model's encoder.
# (Assuming the plddt_projection layer exists and its weight's dtype is what is expected.)
plddt_dtype = loaded_backbone_model.encoder.plddt_projection.weight.dtype

# Create dummy average_plddt and per_res_plddt tensors of the same shape as the sequence tokens.
dummy_average_plddt = torch.zeros(seq_tokens_tensor.shape, dtype=plddt_dtype, device=device)
dummy_per_res_plddt = torch.zeros(seq_tokens_tensor.shape, dtype=plddt_dtype, device=device)
# Create dummy structure tokens (an integer tensor, here zeros)
dummy_structure_tokens = torch.zeros(seq_tokens_tensor.shape, dtype=torch.int64, device=device)
#####


print(f"""dtypes are: \n dummy_average_plddt: {dummy_average_plddt.dtype} 
      \n dummy_per_res_plddt: {dummy_per_res_plddt.dtype} 
      \n dummy_structure_tokens" {dummy_structure_tokens}""")
# dummy_average_plddt, dummy_per_res_plddt, dummy_structure_tokens = None, None, None




# Run the forward pass on the sequence.
with torch.no_grad():
    seq_output = loaded_backbone_model.forward(
         sequence_tokens=seq_tokens_tensor,
         structure_tokens=dummy_structure_tokens,
         average_plddt=dummy_average_plddt,
         per_res_plddt=dummy_per_res_plddt,
         ss8_tokens=None,
         sasa_tokens=None,
         function_tokens=None,
         residue_annotation_tokens=None,
         chain_id=None,
         sequence_id=None
    )

# Inspect the keys of the output.
print("Sequence output keys:", seq_output.keys())

# If the output contains 'representations', extract the last hidden layer's embeddings.
if "representations" in seq_output:
    seq_embeddings = seq_output["representations"][-1]
else:
    # Otherwise, assume the output is directly the embeddings.
    seq_embeddings = seq_output

print("Sequence input test:")
print("  Input sequence:", test_sequence)
print("  Sequence embeddings shape:", seq_embeddings.shape)

