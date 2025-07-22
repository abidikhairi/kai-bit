from torch import FloatTensor, LongTensor, Tensor
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.processing_utils import Unpack


class TextLlamaDecoder(LlamaForCausalLM):
    """
    A text decoder based on the Llama model from Hugging Face Transformers.
    
    This class extends the LlamaForCausalLM to provide a specialized text decoding
    functionality, suitable for tasks that require generating text sequences.
    """
    
    
    def forward(
        self,
        protein_features: FloatTensor | None = None,
        input_ids: LongTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: FloatTensor | None = None,
        labels: LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: LongTensor | None = None,
        logits_to_keep: int | Tensor = 0, 
        **kwargs
    ) -> CausalLMOutputWithPast:
        
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        else:
            raise ValueError("input_ids must be provided if inputs_embeds is not given")
        
        if protein_features is not None:
            assert inputs_embeds is not None, "inputs_embeds must be provided if protein_features are given"
            assert self.config.protein_token_id is not None, "protein_token_id must be set in the model config"
            
            special_protein_mask = input_ids == self.config.protein_token_id
            special_protein_mask = special_protein_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            
            protein_features = protein_features.to(inputs_embeds.device, inputs_embeds.dtype) # type: ignore
            inputs_embeds = inputs_embeds.masked_scatter(special_protein_mask, protein_features)         # type: ignore

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs
        )
