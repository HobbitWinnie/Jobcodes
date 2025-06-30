import os
import numpy as np
import torch
from typing import List
from semantic_matcher.base_semantic_matcher import BaseSemanticMatcher

# CN-CLIP中文多模态封装(这个精度很低，先不用了)
try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name
except ImportError:
    clip, load_from_name = None, None

class CNCLIPSemanticMatcher(BaseSemanticMatcher):
    def __init__(self, model_name: str = 'ViT-B-16', device: str = None, download_root: str = './'):
        assert clip is not None and load_from_name is not None, "请先安装 cn_clip: pip install cn_clip"
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        cache_path = os.path.expanduser('~/.cache/cn_clip')
        os.makedirs(cache_path, exist_ok=True)
        self.model, self.preprocess = load_from_name(model_name, device=self.device, download_root=cache_path)
        self.model.eval()
        self.prompt_fmt = "{}"

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        prompt_texts = [self.prompt_fmt.format(t) for t in texts]
        tokens = clip.tokenize(prompt_texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features.cpu().numpy()
