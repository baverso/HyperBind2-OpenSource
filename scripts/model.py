import torch
import torch.nn as nn
import esm  # Make sure you have cloned the ESM repository and installed its requirements


# Define a classifier model that wraps ESM-3 and adds a classifier head.
class ESM3Classifier(nn.Module):
    def __init__(self, esm_model, embedding_dim, num_classes, hidden_dim=256, dropout=0.1, fine_tune_esm=False):
        super(ESM3Classifier, self).__init__()
        self.esm_model = esm_model
        self.fine_tune_esm = fine_tune_esm

        # Optionally freeze the ESM backbone if you don't wish to fine-tune it.
        if not self.fine_tune_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False

        # Define a classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, tokens):
        # Obtain representations from ESM-3.
        # Here we request the representation from the final layer (repr_layers=[-1]).
        results = self.esm_model(tokens, repr_layers=[-1], return_contacts=False)
        # The token representations shape is (batch_size, sequence_length, embedding_dim)
        token_representations = results["representations"][-1]

        # Pool across the sequence dimension. In this example, we use mean pooling.
        pooled_representation = token_representations.mean(dim=1)

        # Pass the pooled representation through the classifier head.
        logits = self.classifier(pooled_representation)
        return logits


# ------------------------------------------------------------------------------
# Example Usage:
# ------------------------------------------------------------------------------

# 1. Load a pre-trained ESM-3 model.
# Replace "esm3-open" with the appropriate model identifier for your use-case.
# Note: You need access to the model weights.
esm_model, alphabet = esm.pretrained.esm3_tXX_UR50D()  # adjust model identifier as needed
batch_converter = alphabet.get_batch_converter()

# 2. Prepare your data.
# Each entry is a tuple of (name, amino acid sequence)
data = [
    ("protein1", "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQ"),
    ("protein2", "MNKLLVLSVVMLVLLFTVLS"),
]

# The batch converter will tokenize the sequences for the model.
labels, strs, toks = batch_converter(data)
tokens = toks.to("cuda")  # Move tokens to GPU if available

# 3. Create an instance of your classifier.
# Here, `embedding_dim` should match the hidden size of the ESM-3 model.
# For instance, if the ESM-3 model produces embeddings of size 1280:
embedding_dim = 1280  # adjust based on your specific model's architecture
num_classes = 2  # binary classification example
model = ESM3Classifier(esm_model, embedding_dim, num_classes, fine_tune_esm=False).to("cuda")

# 4. Run a forward pass.
logits = model(tokens)
print("Logits:", logits)