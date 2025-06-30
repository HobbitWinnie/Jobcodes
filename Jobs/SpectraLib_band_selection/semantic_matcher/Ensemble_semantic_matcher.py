import sys  
sys.path.append('/home/nw/Codes/Jobs/SpectraLib_band_selection')  

import numpy as np
from typing import List, Dict, Tuple, Any
from utils import pretty_print_clip_match
from semantic_matcher.base_semantic_matcher import BaseSemanticMatcher
from semantic_matcher.CN_CLIP_Matcher import CNCLIPSemanticMatcher
from semantic_matcher.EVA_CLIP_matcher import EVACLIPSemanticMatcher
from semantic_matcher.Text2Vec_Matcher import Text2VecSemanticMatcher


# 集成封装
class EnsembleSemanticMatcher(BaseSemanticMatcher):
    def __init__(self,
                 matcher1: BaseSemanticMatcher,
                 matcher2: BaseSemanticMatcher,
                 mode: str = 'mean'  # 'mean', 'max'                 
                ):
        """
        mode:
            'mean': 三个模型sim结果直接平均
            'max' ：每个分数取最大
        """
        self.matcher1 = matcher1
        self.matcher2 = matcher2
        assert mode in ('mean', 'max'), "mode只支持'mean'或'max'"
        self.mode = mode

    def match(self, query_list: List[str], label_list: List[str], topk: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        # 三个模型分别算embedding
        q_emb1, l_emb1 = self.matcher1.encode_texts(query_list), self.matcher1.encode_texts(label_list)
        q_emb2, l_emb2 = self.matcher2.encode_texts(query_list), self.matcher2.encode_texts(label_list)

        # 归一化
        def norm(x): return x / np.linalg.norm(x, axis=1, keepdims=True)
        q_emb1, l_emb1 = norm(q_emb1), norm(l_emb1)
        q_emb2, l_emb2 = norm(q_emb2), norm(l_emb2)

        # 分别计算余弦相似度
        sim1 = np.dot(q_emb1, l_emb1.T)
        sim2 = np.dot(q_emb2, l_emb2.T)

        # 融合
        if self.mode == 'mean':
            sim = (sim1 + sim2) / 2
        elif self.mode == 'max':
            sim = np.maximum.reduce([sim1, sim2])
        else:
            raise ValueError("不支持的集成模式")

        # 后处理排序输出
        results = {}
        for i, q in enumerate(query_list):
            order = np.argsort(-sim[i])
            matched = [(label_list[idx], float(sim[i, idx])) for idx in order[:topk]]
            results[q] = matched
        return results


# 统一测试示例
def main():
    queries = ["小花猫", "半挂"]
    labels = ["小猫", "小狗", "母猫", "公猫", "猫猫", "航天飞机", "房子", "狗", '大狗', '柴犬']

    print("=== CN-CLIP ===")
    try:
        matcher1 = CNCLIPSemanticMatcher()
        pretty_print_clip_match(matcher1.match(queries, labels, topk=3))
    except Exception as e:
        print(f"CN-CLIP 加载出错: {e}")

    print("=== EVA-CLIP ===")
    try:
        matcher2 = EVACLIPSemanticMatcher()
        pretty_print_clip_match(matcher2.match(queries, labels, topk=3))
    except Exception as e:
        print(f"EVA-CLIP 加载出错: {e}")

    print("=== text2vec-base-chinese ===")
    try:
        matcher3 = Text2VecSemanticMatcher()
        pretty_print_clip_match(matcher3.match(queries, labels, topk=3))
    except Exception as e:
        print(f"text2vec 加载出错: {e}")

    print("=== 融合结果（集成）===")
    try:
        ensemble_matcher = EnsembleSemanticMatcher(matcher2, matcher3, mode='mean')
        pretty_print_clip_match(ensemble_matcher.match(queries, labels, topk=3))
    except Exception as e:
        print(f"Ensemble 加载出错: {e}")

if __name__ == "__main__":
    main()