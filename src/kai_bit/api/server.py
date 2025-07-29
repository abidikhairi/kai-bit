import argparse

from litserve import LitAPI, LitServer
from litserve.loops.base import LitLoop
from litserve.mcp import MCP
from litserve.specs.base import LitSpec
import torch

from kai_bit.config import (
    ProjectionConfig,
    OptimizerConfig,
    TrainingMode
)
from kai_bit.systems import BiologicalInstructionTuning
from kai_bit.utils.object_utils import load_object_from_json


torch.serialization.add_safe_globals([ProjectionConfig, OptimizerConfig, TrainingMode])


class BitAPI(LitAPI):
    
    def __init__(self, ckpt_path: str, projection_config_path: str, max_batch_size: int = 1, batch_timeout: float = 0, api_path: str = "/predict", stream: bool = False, loop: str | LitLoop | None = "auto", spec: LitSpec | None = None, mcp: MCP | None = None, enable_async: bool = False):
        super().__init__(max_batch_size, batch_timeout, api_path, stream, loop, spec, mcp, enable_async)

        optimizer_config: OptimizerConfig = OptimizerConfig()
        projection_config: ProjectionConfig = load_object_from_json(projection_config_path, ProjectionConfig)
        
        self.model = BiologicalInstructionTuning(
            language_model_or_path='data/tmp/language-model',
            protein_encoder_or_path='data/tmp/protein-encoder',
            language_tokenizer_or_path='data/tmp/language-model',
            protein_tokenizer_or_path='data/tmp/protein-encoder',
            projection_config=projection_config,
            optimizer_config=optimizer_config   
        )
        
        self.model.state_dict(torch.load(ckpt_path, weights_only=True)) # type: ignore
        
    def setup(self, device):
        self.model.to(device)        
        self.model.eval()
        
    def decode_request(self, request, **kwargs):
        assert 'protein' in request, "protein should be present"
        assert 'text' in request, "text should be here"
        return super().decode_request(request, **kwargs)
    
    def predict(self, inputs, **kwargs):
        
        return self.model.predict_step(
            protein=inputs['protein'],
            user_input=inputs['text']
        )
        
    def encode_response(self, output, **kwargs):
        return {
            "response": output
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--projection_config_path', default='data/tmp/configs/projection_config.json')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=8088)
    
    args = parser.parse_args()
    
    api = BitAPI(ckpt_path=args.ckpt_path, projection_config_path=args.projection_config_path)
    
    server = LitServer(lit_api=api, accelerator='auto', devices='auto', workers_per_device=2)
    
    server.run(host=args.host, port=args.port)
