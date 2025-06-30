import sys  
sys.path.append('/home/nw/Codes/Jobs/SpectraLib_band_selection')  

import os
import numpy as np
import torch
from typing import List
from semantic_matcher.base_semantic_matcher import BaseSemanticMatcher


# EVA-CLIP多模态大模型
try:
    from eva_clip import create_model_and_transforms, get_tokenizer
except ImportError:
    create_model_and_transforms, get_tokenizer = None, None

class EVACLIPSemanticMatcher(BaseSemanticMatcher):
    def __init__(
        self,
        model_name: str = 'EVA02-CLIP-B-16',
        pretrained: str = 'eva02_clip',
        device: str = None,
        download_root: str = './eva_model_cache'
    ):
        if create_model_and_transforms is None or get_tokenizer is None:
            raise ImportError("请先安装eva_clip相关依赖或添加路径。")
        cache_path = os.path.expanduser('~/.cache/eva_model_cache')
        os.makedirs(cache_path, exist_ok=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=cache_path
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = get_tokenizer(model_name)
        self.prompt_fmt = "{}"

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        prompt_texts = [self.prompt_fmt.format(t) for t in texts]
        tokens = self.tokenizer(prompt_texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features.cpu().numpy()