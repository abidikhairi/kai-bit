from torch import nn
from transformers.models.esm import EsmModel


class EsmProteinEncoder(EsmModel):
    """
    A protein encoder based on the ESM model from Hugging Face Transformers.
    
    This class extends the EsmModel to provide a specialized protein encoding
    functionality.
    """
    pass

class ProteinToTextProjection(nn.Module):
    """
    A projection layer that maps protein embeddings to text embeddings.
    
    This module is designed to project the output of a protein encoder to a
    text embedding space, facilitating tasks that require alignment between
    protein and text representations.
    """
    
    def __init__(
        self,
        protein_hidden_size: int,
        text_hidden_size: int
    ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(protein_hidden_size)
        self.projection = nn.Linear(protein_hidden_size, text_hidden_size, False)

    def forward(self, x):
        x = self.layer_norm(x)
        
        return self.projection(x)
