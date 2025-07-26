import argparse
from datasets import Dataset
import torch
from torchmetrics.functional import translation_edit_rate
from tqdm import tqdm
import pandas as pd

from kai_bit.systems import BiologicalInstructionTuning


torch.set_float32_matmul_precision('medium')


def main(args):
    dataset = Dataset.from_csv(args.input_file)
    model = BiologicalInstructionTuning.load_from_checkpoint(args.ckpt_path).eval()
    
    data = []
    
    for row in tqdm(dataset):
        protein = row['protein'] # type: ignore
        question = row['prompt'] # type: ignore
        
        output = model.predict_step(protein=protein, user_input=question)
        
        predicted_label = output.split('<|annotation_end|>')[0] # type: ignore
        predicted_label = predicted_label.strip()
        label = row['answer'].strip()
        
        ter = translation_edit_rate([predicted_label], [label]).item() # type: ignore
        
        data.append({
            "sequence": protein,
            "label": label,
            "prediction": predicted_label,
            "translation_edit_rate": ter
        })

    result_df = pd.DataFrame(data)

    result_df.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, required=True, help='Path to test file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output file.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint.')
    
    args = parser.parse_args()
    
    main(args)
