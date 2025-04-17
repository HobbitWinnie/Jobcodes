import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any
from ..core.base import BaseCLIPClassifier

class RFClassifier(BaseCLIPClassifier):
    def __init__(self, *args, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.rf = None
        self.classes = []
        self.label_to_index = {}

    def train(self, dataloader, **kwargs):
        features, labels = [], []
        for img_batch, label_batch in dataloader:
            feat = self._get_image_features(img_batch).cpu().numpy()
            features.append(feat)
            labels.extend(label_batch)
        self.classes = sorted(set(labels))
        self.label_to_index = {l: i for i, l in enumerate(self.classes)}
        X = np.vstack(features)
        y = np.array([self.label_to_index[l] for l in labels])
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.rf.fit(X, y)

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
        if self.rf is None:
            raise RuntimeError("Classifier not trained")
        image = self._load_image(img_path)
        
        img_tensor = self.preprocess_func(image).unsqueeze(0)
        features = self._get_image_features(img_tensor).cpu().numpy()
        probs = self.rf.predict_proba(features)[0]
        sorted_indices = np.argsort(probs)[::-1][:3]
        return {
            'label': self.classes[sorted_indices[0]],
            'top3': [(self.classes[i], float(probs[i])) for i in sorted_indices]
        }