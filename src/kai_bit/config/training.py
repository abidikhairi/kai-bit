from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer.
    """
    lm_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Language model learning rate."}
    )
    
    encoder_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Protein encoder learning rate."}
    )
    
    projector_learning_rate: float = field(
        default=1e-3,
        metadata={"help": "Protein projector learning rate."}
    )
    
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "The weight decay for the optimizer."}
    )
    
    betas: tuple = field(
        default=(0.9, 0.98),
        metadata={"help": "The betas for the Adam optimizer."}
    )
    
    eps: float = field(
        default=1e-8,
        metadata={"help": "The epsilon value for the Adam optimizer."}
    )
    
    warmup_steps: int = field(
        default=0,
        metadata={"help": "The number of warmup steps for the learning rate scheduler."}
    )
    
    total_steps: int = field(
        default=10000,
        metadata={"help": "The total number of training steps."}
    )

@dataclass
class TrainingMode:
    """
    Configuration for the training mode.
    """    
    freeze_protein_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the protein encoder during training."}
    )
    
    freeze_protein_pooler: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the protein pooler during training."}
    )
    
    freeze_text_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the text decoder during training."}
    )
    
    freeze_projection_layer: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the projection layer during training."}
    )
