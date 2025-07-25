.PHONY: train-tiny-local

dry-run:
	python scripts/train.py --dry_run --num_workers 0

train-tiny-local:
	python scripts/train.py --num_workers 4 --wandb --batch_size 12 --gradient_accumulation_steps 2 

