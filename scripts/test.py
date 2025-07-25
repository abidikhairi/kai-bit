import argparse
import os
import torch
from pytorch_lightning import (
    Trainer,
    loggers,
    callbacks
)

from kai_bit.systems import BiologicalInstructionTuning
from kai_bit.config import (
    ProjectionConfig,
    OptimizerConfig,
    TrainingMode
)
from kai_bit.datasets import CsvDataModule
from kai_bit.utils import load_object_from_json

torch.set_float32_matmul_precision('medium')


def main(args):
    # MKHNPLVVCLLIICITILTFTLLTRQTLYELRFRDGDKEVAALMACTSR,What is the most probable family for this protein?,Hok/gef cell toxic protein
    # What is the most probable family for this protein?
    # Hok/gef cell toxic protein
    protein = 'MKHNPLVVCLLIICITILTFTLLTRQTLYELRFRDGDKEVAALMACTSR'
    question = 'From its amino acid pattern, which family is implied?'
    
    model = BiologicalInstructionTuning.load_from_checkpoint(args.ckpt_path).eval()
    
    output = model.predict_step(protein=protein, user_input=question)

    print(f"Prediction:\n{output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    
    args = parser.parse_args()
    
    main(args)
