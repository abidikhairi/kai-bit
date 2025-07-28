import argparse
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd


def split_to_dataframe(dataset: Dataset):
    data = []
    
    for row in tqdm(dataset):
        question = row['question'] # type: ignore
        aspect = row['annotation'] # type: ignore
        answer = row['answer'] # type: ignore
        
        if '<protein>' in question:
            chunks = question.split('\n')
            protein: str = chunks[0]
            question: str = chunks[1]
            
            protein = protein.replace('<protein>', '')
            protein = protein.replace('</protein>', '')
            protein = protein.replace(' ', '')
            
            if 'unknown' in answer:
                continue
                
            data.append({
                'protein': protein,
                'prompt': question,
                'answer': answer
            })
        
    return pd.DataFrame(data)


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    ds = load_dataset(args.dataset_id)

    for split in ['train', 'validation', 'test']:
        df = split_to_dataframe(ds[split]).sample(frac=1.0) # type: ignore

        df.to_csv(f'{args.save_dir}/{split}.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_id', default='khairi/SwissProt-QA-Small', help='Huggingface dataset id.')
    parser.add_argument('--save_dir', required=True, help='Data save directory.')
    
    args = parser.parse_args()
    main(args)