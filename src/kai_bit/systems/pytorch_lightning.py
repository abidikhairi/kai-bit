from typing import Union, Optional
import millify
from torch import LongTensor, optim
import torch
from transformers import AutoTokenizer
from transformers.trainer_utils import SchedulerType
from transformers.optimization import get_scheduler
from pytorch_lightning import LightningModule

from kai_bit.modeling import (
    TextLlamaDecoder,
    EsmProteinEncoder,
    ProteinToTextProjection
)

from kai_bit.config import (
    ProjectionConfig,
    OptimizerConfig,
    TrainingMode
)


class BiologicalInstructionTuning(LightningModule):
    def __init__(
        self, 
        language_model_or_path: Union[str, TextLlamaDecoder],
        protein_encoder_or_path: Union[str, EsmProteinEncoder],
        language_tokenizer_or_path: Union[str, AutoTokenizer],
        protein_tokenizer_or_path: Union[str, AutoTokenizer],
        projection_config: ProjectionConfig,
        optimizer_config: OptimizerConfig,
        training_mode: TrainingMode = TrainingMode(
            freeze_projection_layer=False,
            freeze_protein_encoder=True,
            freeze_protein_pooler=True,
            freeze_text_decoder=False
        ),
        **kwargs
    ):

        super().__init__(**kwargs)
        
        self.projection_config = projection_config
        self.optimizer_config = optimizer_config
        self.training_mode = training_mode
        
        self.save_hyperparameters()

        if isinstance(protein_encoder_or_path, str):
            self.protein_encoder = EsmProteinEncoder.from_pretrained(protein_encoder_or_path)
        elif isinstance(protein_encoder_or_path, EsmProteinEncoder):
            self.protein_encoder = protein_encoder_or_path
        else:
            raise ValueError("protein_encoder_or_path must be a string or an instance of EsmProteinEncoder")
        
        self.prot2text = ProteinToTextProjection(
            self.projection_config.protein_hidden_size,
            self.projection_config.text_hidden_size,
        )
        
        if isinstance(language_model_or_path, str):
            self.language_model = TextLlamaDecoder.from_pretrained(language_model_or_path)
        elif isinstance(language_model_or_path, TextLlamaDecoder):
            self.language_model = language_model_or_path
        else:
            raise ValueError("language_model_or_path must be a string or an instance of TextLlamaDecoder")

        self.language_model = self.language_model.train()
        self.protein_encoder = self.protein_encoder.train()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        if isinstance(language_tokenizer_or_path, str):
            self.language_tokenizer = AutoTokenizer.from_pretrained(language_tokenizer_or_path)
        elif isinstance(language_tokenizer_or_path, AutoTokenizer):
            self.language_tokenizer = language_tokenizer_or_path
        else:
            raise ValueError("language_tokenizer_or_path must be a string or an instance of AutoTokenizer")
        
        if isinstance(protein_tokenizer_or_path, str):
            self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_tokenizer_or_path)
        elif isinstance(protein_tokenizer_or_path, AutoTokenizer):
            self.protein_tokenizer = protein_tokenizer_or_path
        else:
            raise ValueError("protein_tokenizer_or_path must be a string or an instance of AutoTokenizer")
            
        if self.training_mode.freeze_protein_encoder:
            self.protein_encoder.encoder.requires_grad_(False)
                
        if self.training_mode.freeze_protein_pooler:
            self.protein_encoder.pooler.requires_grad_(False) # type: ignore

        if self.training_mode.freeze_text_decoder:
            self.language_model.requires_grad_(False)
                
        if self.training_mode.freeze_projection_layer:
            self.prot2text.requires_grad_(False)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [
                {
                    "params": self.protein_encoder.parameters(),
                    'lr': self.optimizer_config.encoder_learning_rate,
                    "name": "protein_encoder"
                },
                {
                    "params": self.prot2text.parameters(),
                    'lr': self.optimizer_config.projector_learning_rate,
                    "name": "protein_projector"
                },
                {
                    'params': self.language_model.parameters(),
                    'lr': self.optimizer_config.lm_learning_rate,
                    "name": "language_model"
                }
            ],
            eps=self.optimizer_config.eps,
            weight_decay=self.optimizer_config.weight_decay,
            betas=self.optimizer_config.betas,
        )
        lr_scheduler = get_scheduler(
            name=SchedulerType.LINEAR,
            optimizer=optimizer,
            num_warmup_steps=self.optimizer_config.warmup_steps,
            num_training_steps=self.optimizer_config.total_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": None,
            }
        }

    def _get_protein_features(
        self,
        protein_input_ids: LongTensor,
        protein_attention_mask: Optional[LongTensor] = None, # type: ignore
    ):
        
        protein_features = self.protein_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask
        ).pooler_output
        
        return self.prot2text(protein_features)

    def get_inputs_embeddings(
        self,
        protein_input_ids: LongTensor,
        input_ids: LongTensor,
        protein_attention_mask: Optional[LongTensor] = None, # type: ignore
        attention_mask: Optional[LongTensor] = None, # type: ignore
    ):
        protein_input_ids = protein_input_ids.to(self.device) # type: ignore
        protein_attention_mask = protein_attention_mask.to(self.device) # type: ignore
        
        input_ids = input_ids.to(self.device) # type: ignore
        attention_mask = attention_mask.to(self.device) # type: ignore
        
        protein_features = self._get_protein_features(
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask
        )
        
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        special_protein_mask = input_ids == self.language_model.config.protein_token_id
        special_protein_mask = special_protein_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            
        protein_features = protein_features.to(inputs_embeds.device, inputs_embeds.dtype) # type: ignore
        inputs_embeds = inputs_embeds.masked_scatter(special_protein_mask, protein_features)         # type: ignore

        return inputs_embeds

    def forward(
        self,
        protein_input_ids: LongTensor,
        input_ids: LongTensor,
        protein_attention_mask: Optional[LongTensor] = None, # type: ignore
        attention_mask: Optional[LongTensor] = None, # type: ignore
        labels: Optional[LongTensor] = None, # type: ignore
        **kwargs
    ):
        """
        Forward pass of the model.
        Args:
            protein_input_ids: Input IDs for the protein sequences.
            input_ids: Input IDs for the text sequences.
            protein_attention_mask: Attention mask for the protein sequences.
            attention_mask: Attention mask for the text sequences.
            labels: Labels for the text sequences.
            **kwargs: Additional keyword arguments.
        Returns:
            The output of the model.
        """
        protein_features = self._get_protein_features(
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask
        )
        
        return self.language_model(
            protein_features=protein_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.
        
        Returns:
            The loss value for the training step.
        """
        protein_input_ids = batch['protein_input_ids']
        protein_attention_mask = batch['protein_attention_mask']

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
    
        outputs = self(
            protein_input_ids=protein_input_ids,
            input_ids=input_ids,
            protein_attention_mask=protein_attention_mask,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        
        logits = logits.view(-1, self.language_model.config.vocab_size)
        
        loss = self.loss_fn(logits, labels.view(-1))        
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/perplexity", loss.exp(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True, sync_dist=True) # type: ignore
        
        return loss

    
    def validation_step(self, batch, batch_idx):
        """
        Override this method to define the validation step.
        
        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.
        
        Returns:
            The loss value for the validation step.
        """
        
        protein_input_ids = batch['protein_input_ids']
        protein_attention_mask = batch['protein_attention_mask']

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
    
        outputs = self(
            protein_input_ids=protein_input_ids,
            input_ids=input_ids,
            protein_attention_mask=protein_attention_mask,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        
        logits = logits.view(-1, self.language_model.config.vocab_size)
        
        loss = self.loss_fn(logits, labels.view(-1))        
        
        self.log("valid/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("valid/perplexity", loss.exp(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Override this method to define the test step.
        
        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.
        
        Returns:
            The loss value for the test step.
        """
        return super().test_step(batch, batch_idx)

    def _zeros(self, n: int):
        return [0] * n
    
    def predict_step(self, protein: str, user_input: str):
        assistant_response = {}
        text_input = {}
        
        user_message_start = self.language_tokenizer("<|im_start|>user\n") # type: ignore
        user_message_end = self.language_tokenizer("\n<|im_end|>") # type: ignore
        assistant_message_start = self.language_tokenizer("<|im_start|>assistant\n") # type: ignore
        
        protein_inputs = self.protein_tokenizer(protein, return_tensors='pt') # type: ignore
        
        user_input = self.language_tokenizer(f'{self.language_tokenizer.protein_token} {user_input}') # type: ignore
        
        user_input['input_ids'] = user_message_start['input_ids'] + user_input['input_ids'] + user_message_end['input_ids'] # type: ignore
        user_input['attention_mask'] = self._zeros(len(user_message_start['input_ids'])) + user_input['attention_mask'] + self._zeros(len(user_message_end['input_ids'])) # type: ignore

        assistant_response['input_ids'] = assistant_message_start['input_ids']
        assistant_response['attention_mask'] = self._zeros(len(assistant_message_start['input_ids']))     # type: ignore
        
        text_input['input_ids'] = user_input['input_ids'] + assistant_response['input_ids'] # type: ignore
        text_input['attention_mask'] = user_input['attention_mask'] + assistant_response['attention_mask'] # type: ignore

        text_input['input_ids'] = torch.tensor(text_input['input_ids']).long()
        text_input['attention_mask'] = torch.tensor(text_input['attention_mask']).long()
        
        inputs_embeds = self.get_inputs_embeddings(
            protein_input_ids=protein_inputs['input_ids'],
            protein_attention_mask=protein_inputs['attention_mask'],
            input_ids=text_input['input_ids'], # type: ignore
            attention_mask=text_input['attention_mask'] # type: ignore
        )
        attention_mask = text_input['attention_mask'].unsqueeze(0).to(self.device)
        
        inputs_embeds = inputs_embeds.unsqueeze(0).to(self.device)
        
        output = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=32,
            # TODO: fix generation parameters
            # do_sample=True,
            # temperature=0.7,
            # top_p=0.9,
            # top_k=50,
            eos_token_id=self.language_tokenizer.eos_token_id, # type: ignore
            pad_token_id=self.language_tokenizer.pad_token_id # type: ignore
        )
        
        return self.language_tokenizer.decode(output[0]) # type: ignore
