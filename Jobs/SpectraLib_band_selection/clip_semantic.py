# clip_semantic.py

import torch
import clip
import numpy as np

class CLIPSemanticMatcher:
    def __init__(self, device='cpu'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def encode_texts(self, texts):
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens).cpu().numpy()  # (n, d)

    def match(self, query_list, label_list, topk=2, threshold=0.3):
        """
        query_list: 用户输入的目标语义（如["树木", "水体"]）
        label_list: 光谱库中所有可选标签（如["柏树", "杨树", "农田"]）
        返回：dict，key为query，每个value为label索引和分数
        """
        q_emb = self.encode_texts(query_list)  # (m, d)
        l_emb = self.encode_texts(label_list)  # (n, d)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        l_emb = l_emb / np.linalg.norm(l_emb, axis=1, keepdims=True)
        sim = np.matmul(q_emb, l_emb.T)  # (m, n)
        results = {}
        for i, q in enumerate(query_list):
            order = np.argsort(-sim[i])  # 从大到小
            matched = []
            for idx in order:
                if sim[i, idx] >= threshold:
                    matched.append((label_list[idx], float(sim[i, idx])))
                if len(matched) >= topk:
                    break
            results[q] = matched
        return results