import numpy as np
import torch
from typing import List
from semantic_matcher.base_semantic_matcher import BaseSemanticMatcher


##############################
# text2vec-base-chinese 句向量模型
##############################
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer, AutoModel = None, None

class Text2VecSemanticMatcher(BaseSemanticMatcher):
    def __init__(
        self,
        model_name: str = 'GanymedeNil/text2vec-base-chinese',
        device: str = None
    ):
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("请先安装 transformers")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.prompt_fmt = "{}"

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        prompt_texts = [self.prompt_fmt.format(t) for t in texts]
        inputs = self.tokenizer(
            prompt_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            embeddings = outputs.last_hidden_state[:, 0]
        return embeddings.cpu().numpy()