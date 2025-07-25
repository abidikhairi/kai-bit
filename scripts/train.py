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
    protein_tokenizer_path = args.protein_tokenizer_path
    language_tokenizer_path = args.language_tokenizer_path
    
    optimizer_config: OptimizerConfig = load_object_from_json(args.optimizer_config_path, OptimizerConfig)
    training_mode: TrainingMode = load_object_from_json(args.training_mode_path, TrainingMode)
    projection_config: ProjectionConfig = load_object_from_json(args.projection_config_path, ProjectionConfig)
        
    datamodule = CsvDataModule(
        base_dir=args.base_dir,
        protein_tokenizer_or_path=protein_tokenizer_path,
        language_tokenizer_or_path=language_tokenizer_path,
        protein_column=args.protein_column,
        prompt_column=args.prompt_column,
        response_column=args.response_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    datamodule.setup()
    
    model = BiologicalInstructionTuning(
        language_model_or_path=args.language_model_path,
        protein_encoder_or_path=args.protein_encoder_path,
        language_tokenizer_or_path=language_tokenizer_path,
        protein_tokenizer_or_path=protein_tokenizer_path,
        optimizer_config=optimizer_config,
        projection_config=projection_config,
        training_mode=training_mode
    )
    
    logger = []
    
    logger.append(loggers.CSVLogger(
            save_dir=args.experim_dir,
            name='logs',
            flush_logs_every_n_steps=100
        )
    )
    
    if args.wandb:
        logger.append(loggers.WandbLogger(
                name=os.getenv('WANDB_NAME', 'bit-tiny-0725'),
                save_dir=f'{args.experim_dir}/wandb',
                project=os.getenv('WANDB_PROJECT', 'BIT')
            )
        )
    
    trainer = Trainer(
        accelerator='auto',
        log_every_n_steps=100,
        max_epochs=1,
        # max_steps=optimizer_config.total_steps,
        overfit_batches=4 if args.dry_run else 0,
        default_root_dir=args.experim_dir,
        max_time='00:08:00:00', # training should not last more than 8 hours
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[
            callbacks.ModelCheckpoint(
                dirpath=f'{args.experim_dir}/checkpoints',
                filename='bit-v1-{step:05d}',
                monitor='train/loss',
                save_top_k=3,
                mode='min',
                every_n_train_steps=100
            ),
            callbacks.EarlyStopping(
                monitor='valid/loss',
                patience=3
            ),
            callbacks.LearningRateMonitor(
                logging_interval='step'
            )
        ],
        logger=logger
    )
    
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_dir', type=str, default='data/tmp/datasets/uniprot')
    parser.add_argument('--protein_tokenizer_path', type=str, default='data/tmp/protein-encoder')
    parser.add_argument('--language_tokenizer_path', type=str, default='data/tmp/language-model')
    parser.add_argument('--protein_column', type=str, default='protein')
    parser.add_argument('--prompt_column', type=str, default='prompt')
    parser.add_argument('--response_column', type=str, default='answer')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0) 
    
    parser.add_argument('--optimizer_config_path', type=str, default='data/tmp/configs/optimizer_config.json')
    parser.add_argument('--training_mode_path', type=str, default='data/tmp/configs/training_mode.json')
    parser.add_argument('--projection_config_path', type=str, default='data/tmp/configs/projection_config.json')

    parser.add_argument('--protein_encoder_path', type=str, default='data/tmp/protein-encoder')
    parser.add_argument('--language_model_path', type=str, default='data/tmp/language-model')

    parser.add_argument('--experim_dir', type=str, default="data/tmp/experiment/bit-v1")
    parser.add_argument('--dry_run', action='store_true', help='Run without training to test the setup.')
    parser.add_argument('--wandb', action='store_true', help='Wether to use wandb logger or not')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients before backward/update.')
    
    args = parser.parse_args()
    
    main(args)
