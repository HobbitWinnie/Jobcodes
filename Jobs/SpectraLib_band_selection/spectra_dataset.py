import pandas as pd
from semantic_matcher.Ensemble_semantic_matcher import CLIPSemanticMatcher


class SpectraDataset:
    """
    管理光谱数据与标签，支持语义筛选
    data: DataFrame，行是样本，列是波段
    labels: list，每个样本的标签(一级/二级)
    class_map: dict，目标语义到标签映射，如{"树木": ["柏树","杨树"]}
    """

    def __init__(self, data, labels, class_map=None):
        self.data = data    # shape: (samples, bands)
        self.labels = labels # list[str]
        self.class_map = class_map or {}

    @classmethod
    def from_csv(cls, spectrum_path, label_path, class_map_path=None):
        data = pd.read_csv(spectrum_path).values
        labels = pd.read_csv(label_path)["label"].tolist()
        class_map = None
        if class_map_path:
            class_map = pd.read_json(class_map_path).to_dict()
        return cls(data, labels, class_map)

    def get_samples_by_semantic(self, targets):
        """
        按目标语义进行检索，返回包含样本和标签的子集
        """
        resolved_targets = []
        for t in targets:
            resolved_targets += self.class_map.get(t, [t])
        indices = [i for i, l in enumerate(self.labels) if l in resolved_targets]
        sub_data = self.data[indices]
        sub_labels = [self.labels[i] for i in indices]
        return sub_data, sub_labels
    
    def get_samples_by_clip(self, user_targets, topk=2, threshold=0.2, device='cpu'):
        # 自动用CLIP选label
        matcher = CLIPSemanticMatcher(device=device)
        label_set = sorted(list(set(self.labels)))  # 数据库标签全集
        semantic_map = matcher.match(user_targets, label_set, topk=topk, threshold=threshold)
        resolved_labels = []
        for q in user_targets:
            # 支持单列输出
            resolved_labels += [lab for lab, score in semantic_map.get(q, [])]
        indices = [i for i, l in enumerate(self.labels) if l in resolved_labels]
        sub_data = self.data[indices]
        sub_labels = [self.labels[i] for i in indices]
        return sub_data, sub_labels, semantic_map