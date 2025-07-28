experim_dir ?= data/tmp/experiment/bit-v1
gradient_accumulation_steps ?= 1
base_dir ?= data/tmp/datasets/uniprot
protein_tokenizer_path ?= data/tmp/protein-encoder
language_tokenizer_path ?= data/tmp/language-model
protein_encoder_path ?= data/tmp/protein-encoder
language_model_path ?= data/tmp/language-model
protein_column ?= protein
prompt_column ?= prompt
response_column ?= answer
batch_size ?= 2
num_workers ?= 0

.PHONY: train-tiny-local dry-run

dry-run:
	python scripts/train.py --dry_run --num_workers 0

train-tiny-local:
	python scripts/train.py \
		--wandb \
		--base_dir $(base_dir) \
		--experim_dir $(experim_dir) \
		--protein_tokenizer_path $(protein_tokenizer_path) \
		--language_tokenizer_path $(language_tokenizer_path) \
		--protein_encoder_path $(protein_encoder_path) \
		--language_model_path $(language_model_path) \
		--protein_column $(protein_column) \
		--prompt_column $(prompt_column) \
		--response_column $(response_column) \
		--num_workers $(num_workers) \
		--batch_size $(batch_size) \
		--gradient_accumulation_steps $(gradient_accumulation_steps) 

continue-training:
	python scripts/continue_from_checkpoint.py \
		--experim_dir $(experim_dir) \
		--gradient_accumulation_steps $(gradient_accumulation_steps) \
		--base_dir $(base_dir) \
		--protein_tokenizer_path $(protein_tokenizer_path) \
		--language_tokenizer_path $(language_tokenizer_path) \
		--protein_column $(protein_column) \
		--prompt_column $(prompt_column) \
		--response_column $(response_column) \
		--batch_size $(batch_size) \
		--num_workers $(num_workers) \
		--ckpt_path $(ckpt_path)
