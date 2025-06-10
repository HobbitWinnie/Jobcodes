import os
import numpy as np
import torch
from typing import List, Dict, Tuple, Any
from utils import pretty_print_clip_match


##############################
# 基类定义
##############################
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

##############################
# CN-CLIP中文多模态封装(这个精度很低，先不用了)
##############################
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

##############################
# EVA-CLIP多模态大模型
##############################
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

##############################
# 三模型集成封装
##############################
class EnsembleSemanticMatcher(BaseSemanticMatcher):
    def __init__(self,
                 matcher1: BaseSemanticMatcher,
                 matcher2: BaseSemanticMatcher,
                 mode: str = 'mean'  # 'mean', 'max',
                 
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
            sim = (sim1 + sim2) / 3
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

##############################
# 统一测试示例
##############################
def main():
    queries = ["小花猫", "飞机"]
    labels = ["小猫", "小狗", "母猫", "公猫", "猫咪", "航天飞机", "房子", "狗", '大狗', '柴犬']

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
        ensemble_matcher = EnsembleSemanticMatcher(mode='mean')
        pretty_print_clip_match(ensemble_matcher.match(queries, labels, topk=3))
    except Exception as e:
        print(f"Ensemble 加载出错: {e}")

if __name__ == "__main__":
    main()