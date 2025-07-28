import argparse
from transformers.models import (
    EsmModel
)
from transformers import AutoTokenizer

from kai_bit.utils.modeling_utils import fork_llama_model


def main(args):
    protein_tokenizer = AutoTokenizer.from_pretrained(args.protein_encoder_id)
    language_tokenizer = AutoTokenizer.from_pretrained(
        args.language_model_id,
        extra_special_tokens = {
            'protein_token': '<|protein|>',
            'annotation_end_token': '<|annotation_end|>' # used to control when generation stop 
        }
    )
    
    protein_encoder = EsmModel.from_pretrained(args.protein_encoder_id)
    language_model = fork_llama_model(
        model_id=args.language_model_id,
        protein_token_id=language_tokenizer.protein_token_id,
        num_added_tokens=2
    )

    protein_encoder.save_pretrained(args.protein_encoder_path)    
    protein_tokenizer.save_pretrained(args.protein_encoder_path)    
        
    language_model.save_pretrained(args.language_model_path)
    language_tokenizer.save_pretrained(args.language_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--protein_encoder_id', type=str, default='facebook/esm2_t6_8M_UR50D')
    parser.add_argument('--language_model_id', type=str, default='HuggingFaceTB/SmolLM2-135M-Instruct')
    
    parser.add_argument('--protein_encoder_path', type=str, default='/tmp/protein-encoder')
    parser.add_argument('--language_model_path', type=str, default='/tmp/language-model')
    
    args = parser.parse_args()
    main(args)
