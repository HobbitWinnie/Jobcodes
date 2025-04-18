import torch
from typing import List, Dict, Any
from ..core.base import BaseCLIPClassifier

class ZeroShotClassifier(BaseCLIPClassifier):
    def __init__(self, *args, labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels or []
        self.text_features = None
        self._prepare_text_features()

    def _prepare_text_features(self):
        if not self.labels:
            raise ValueError("必须提供标签列表")
        self.text_features = self._get_text_features(self.labels).to(torch.float32)

    def train(self, train_loader=None, val_loader=None, **kwargs):
        self._prepare_text_features()
        
        if val_loader is not None:
            acc = self.evaluate(val_loader)
            self.logger.info(f'Val acc: {acc['accuracy']}')

    def evaluate(self, data_loader) -> Dict:
        correct, total = 0, 0
        for img_batch, label_batch in data_loader:
            for img, gt in zip(img_batch, label_batch):
                pred = self._predict_single(img)
                if pred['label'] == gt:
                    correct += 1
                total += 1
        acc = correct / total if total else 0
        return {'accuracy': acc}

    def _predict_single(self, img_path: str) -> Dict[str, Any]:
        if not self.labels:
            raise RuntimeError("未设置分类标签")
        image = self._load_image(img_path)
        
        img_tensor = self.preprocess_func(image).unsqueeze(0)
        img_features = self._get_image_features(img_tensor)
        similarity = (100.0 * img_features @ self.text_features.T).softmax(dim=-1)
        probs, indices = similarity.topk(3)
        return {
            'label': self.labels[indices[0][0].item()],
            'top3': [(self.labels[i.item()], float(p)) for p, i in zip(probs[0], indices[0])]
        }