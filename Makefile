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
optimizer_config_path ?= data/tmp/configs/local/optimizer_config.json
projection_config_path ?= data/tmp/configs/local/projection_config.json
training_mode_path ?= data/tmp/configs/local/training_mode.json
max_epochs ?= 1

.PHONY: train-tiny-local dry-run continue-training run-server

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
		--max_epochs $(max_epochs) \
		--optimizer_config_path $(optimizer_config_path) \
		--projection_config_path $(projection_config_path) \
		--training_mode_path $(training_mode_path)
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

run-server:
	python src/kai_bit/api/server.py \
		--ckpt_path data/tmp/ckpts/bit-v1-step=14900.ckpt \
		--port 9998
