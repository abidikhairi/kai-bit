from typing import Union, Optional
from torch import LongTensor, optim
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
            for param in self.protein_encoder.encoder.parameters():
                param.requires_grad = False
                
        if self.training_mode.freeze_protein_pooler:
            for param in self.protein_encoder.pooler.parameters(): # type: ignore
                param.requires_grad = False
    
        if self.training_mode.freeze_text_decoder:
            for param in self.language_model.parameters():
                param.requires_grad = False
                
        if self.training_mode.freeze_projection_layer:
            for param in self.prot2text.parameters():
                param.requires_grad = False
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.learning_rate,
            eps=self.optimizer_config.eps,
            weight_decay=self.optimizer_config.weight_decay,
            betas=self.optimizer_config.betas,
        )
        lr_scheduler = get_scheduler(
            name=SchedulerType.CONSTANT_WITH_WARMUP,
            optimizer=optimizer,
            num_warmup_steps=self.optimizer_config.warmup_steps,
            num_training_steps=self.optimizer_config.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }
    

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
        protein_features = self.protein_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask
        ).pooler_output
        
        protein_features = self.prot2text(protein_features)
        
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
        labels = batch['input_ids'].clone()
        
        labels[labels == self.language_tokenizer.pad_token_id] = -100 # type: ignore
        
        outputs = self(
            protein_input_ids=protein_input_ids,
            input_ids=input_ids,
            protein_attention_mask=protein_attention_mask,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        self.log("train_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_perplexity", outputs.loss.exp(), on_step=True, on_epoch=True, logger=True)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True) # type: ignore
        
        return outputs.loss

    
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
        labels = batch['input_ids'].clone()
        
        labels[labels == self.language_tokenizer.pad_token_id] = -100 # type: ignore
        
        outputs = self(
            protein_input_ids=protein_input_ids,
            input_ids=input_ids,
            protein_attention_mask=protein_attention_mask,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        self.log("train_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss", outputs.loss.exp(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return outputs.loss
    
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
    
    def predict_step(self, batch, batch_idx):
        """
        Override this method to define the prediction step.
        
        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.
        
        Returns:
            The predictions for the input batch.
        """
        return super().predict_step(batch, batch_idx)
