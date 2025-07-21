from transformers.models import (
    LlamaConfig
)
from kai_bit.modeling import (
    TextLlamaDecoder
)
from kai_bit.config import (
    TextLlamaConfig
)

def fork_llama_model(
        model_id: str,
        protein_token_id: int
    ):
    """
    Forks the model with the given model_id.
    Copy the model weights and configuration to a new instance.
    
    Args:
        model_id (str): The identifier of the model to fork.
        protein_token_id (int): The protein token ID to set in the configuration.
    
    Returns:
        The forked model instance.
    """
    config = LlamaConfig.from_pretrained(model_id)
    new_config = TextLlamaConfig(**config.to_dict(), protein_token_id=protein_token_id)
    new_config.vocab_size += 1  # Increase vocab size for the new protein token
    
    model = TextLlamaDecoder.from_pretrained(model_id)
    
    embedding_layer = model.resize_token_embeddings(config.vocab_size + 1)
    model.set_input_embeddings(embedding_layer)
    
    model.config = new_config
    
    return model
