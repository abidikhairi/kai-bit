import os.path as osp
from typing import Optional, Union
from datasets import Dataset
from pytorch_lightning import LightningDataModule
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class CsvDataModule(LightningDataModule):
    def __init__(
        self,
        base_dir: str,
        protein_tokenizer_or_path: Optional[Union[str, AutoTokenizer]] = None,
        language_tokenizer_or_path: Optional[Union[str, AutoTokenizer]] = None,
        protein_column: str = 'protein',
        prompt_column: str = 'prompt',
        response_column: str = 'response',
        batch_size: int = 32,
        num_workers: int = 0
    ):
        super().__init__()
        
        if protein_tokenizer_or_path is None:
            raise ValueError("protein_tokenizer_or_path must be provided")
        
        if language_tokenizer_or_path is None:
            raise ValueError("language_tokenizer_or_path must be provided")
        
        if isinstance(protein_tokenizer_or_path, str):
            self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_tokenizer_or_path)
        else:
            self.protein_tokenizer = protein_tokenizer_or_path
            
        if isinstance(language_tokenizer_or_path, str):
            self.language_tokenizer = AutoTokenizer.from_pretrained(language_tokenizer_or_path)
        else:
            self.language_tokenizer = language_tokenizer_or_path

        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.protein_column = protein_column
        self.prompt_column = prompt_column
        self.response_column = response_column
        
        self.dataset_paths = {
            'train': f'{base_dir}/train.csv',
            'validation': f'{base_dir}/validation.csv',
            'test': f'{base_dir}/test.csv'
        }
        
        self.dataset = {}

    def setup(self, stage=None):
        for split, path in self.dataset_paths.items():
            if not osp.exists(path):
                raise FileNotFoundError(f"Dataset file {path} does not exist.")
            self.dataset[split] = self._load_dataset(path)


    def _load_dataset(self, csv_file):
        return Dataset.from_csv(csv_file) \
            .select_columns([self.protein_column, self.prompt_column, self.response_column]) \
            .with_format('torch')

    def _zeros(self, n: int):
        return [0 for _ in range(n)]

    def _collate_fn(self, batch):
        proteins = [item[self.protein_column] for item in batch]
        prompts = [item[self.prompt_column] for item in batch]
        responses = [item[self.response_column] for item in batch]
        
        protein_inputs = self.protein_tokenizer(proteins, padding=True, return_tensors='pt') # type: ignore
        protein_inputs = {f'protein_{k}': v for k, v in protein_inputs.items()}
        
        user_message_start = self.language_tokenizer("<|im_start|>user\n") # type: ignore
        user_message_end = self.language_tokenizer("\n<|im_end|>") # type: ignore
        assistant_message_start = self.language_tokenizer("<|im_start|>assistant\n") # type: ignore
        assistant_message_end = self.language_tokenizer("\n<|im_end|>") # type: ignore
        
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        for p, r in zip(prompts, responses):
            if not p.startswith(self.language_tokenizer.protein_token): # type: ignore
                p = f"{self.language_tokenizer.protein_token} {p}" # type: ignore
         
            user_input = self.language_tokenizer(p) # type: ignore
            user_input['input_ids'] = user_message_start['input_ids'] + user_input['input_ids'] + user_message_end['input_ids'] # type: ignore
            user_input['attention_mask'] = self._zeros(len(user_message_start['input_ids'])) + user_input['attention_mask'] + self._zeros(len(user_message_end['input_ids'])) # type: ignore
            
            assistant_response = self.language_tokenizer(r) # type: ignore
            assistant_response['input_ids'] = assistant_message_start['input_ids'] + assistant_response['input_ids'] + assistant_message_end['input_ids'] # type: ignore
            assistant_response['attention_mask'] = self._zeros(len(assistant_message_start['input_ids'])) + assistant_response['attention_mask'] + self._zeros(len(assistant_message_end['input_ids'])) # type: ignore

            input_ids = user_input['input_ids'] + assistant_response['input_ids']
            attention_mask = user_input['attention_mask'] + assistant_response['attention_mask']
            
            input_ids = torch.tensor(input_ids).long()
            attention_mask = torch.tensor(attention_mask).long()
            
            # labels = self._zeros(len(user_input['input_ids'])) + assistant_response['input_ids']
            
            # train on user inputs + assistant completion
            labels = user_input['input_ids'] + assistant_response['input_ids']
            labels = torch.tensor(labels).long()
            
            labels = labels * attention_mask
            labels[labels == 0] = -100
            labels[labels == self.language_tokenizer.pad_token_id] = -100 # type: ignore
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
        
        all_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.language_tokenizer.pad_token_id) # type: ignore
        all_attention_mask = pad_sequence(all_attention_mask, batch_first=True, padding_value=0) # type: ignore
        all_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100) # type: ignore
        
        batch = {
            **protein_inputs,
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'labels': all_labels
        }
        
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )


    def val_dataloader(self):
        return DataLoader(
            self.dataset['validation'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.dataset['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate_fn
        )
    