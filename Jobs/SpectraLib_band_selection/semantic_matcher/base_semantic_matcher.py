import numpy as np
from typing import List, Dict, Tuple, Any


# 基类定义
class BaseSemanticMatcher:
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def match(self, query_list: List[str], label_list: List[str], topk: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        q_emb = self.encode_texts(query_list)
        l_emb = self.encode_texts(label_list)
        # 归一化
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        l_emb = l_emb / np.linalg.norm(l_emb, axis=1, keepdims=True)
        sim = np.dot(q_emb, l_emb.T)
        results = {}
        for i, q in enumerate(query_list):
            order = np.argsort(-sim[i])
            matched = [(label_list[idx], float(sim[i, idx])) for idx in order[:topk]]
            results[q] = matched
        return results