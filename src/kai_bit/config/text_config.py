from transformers.models.llama import (
    LlamaConfig
)
from transformers.models.qwen2 import (
    Qwen2Config
)


class TextLlamaConfig(LlamaConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.protein_token_id = kwargs.get("protein_token_id", None)
        if self.protein_token_id is not None:
            assert isinstance(self.protein_token_id, int), "protein_token_id must be an integer"
    
class TextQwen2Config(Qwen2Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.protein_token_id = kwargs.get("protein_token_id", None)
        if self.protein_token_id is not None:
            assert isinstance(self.protein_token_id, int), "protein_token_id must be an integer"
            
TEXT_CONFIGS = {
    "llama": TextLlamaConfig,
    "qwen2": TextQwen2Config
}
