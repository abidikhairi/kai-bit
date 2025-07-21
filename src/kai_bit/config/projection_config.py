from dataclasses import dataclass, field

@dataclass
class ProjectionConfig:
    """
    Configuration for the projection layer.
    """
    protein_hidden_size: int = field(
        default=320,
        metadata={"help": "The hidden size of the protein encoder."}
    )
    
    text_hidden_size: int = field(
        default=592,
        metadata={"help": "The hidden size of the text decoder."}
    )