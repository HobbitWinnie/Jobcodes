import numpy as np
import torch
from sklearn.svm import SVC
from typing import Dict, Any
from ..core.base import BaseCLIPClassifier

class SVMClassifier(BaseCLIPClassifier):
    def __init__(self, *args, C=1.0, kernel='linear', **kwargs):
        super().__init__(*args, **kwargs)
        self.C = C
        self.kernel = kernel
        self.svm = None
        self.classes = []
        self.label_to_index = {}
        self.svm = SVC(C=self.C, kernel=self.kernel, probability=True)

    def train(self, train_loader, val_loader=None, **kwargs):
        features, labels = [], []
        for img_batch, label_batch in train_loader:
            feat = self._get_image_features(img_batch).cpu().numpy()
            features.append(feat)
            labels.extend(label_batch)

        self.classes = sorted(set(labels))
        self.label_to_index = {l: i for i, l in enumerate(self.classes)}
        
        X = np.vstack(features)
        y = np.array([self.label_to_index[l] for l in labels])
        self.svm.fit(X, y)

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
        if self.svm is None:
            raise RuntimeError("Classifier not trained")
        image = self._load_image(img_path)
        
        img_tensor = self.preprocess_func(image).unsqueeze(0)
        features = self._get_image_features(img_tensor).cpu().numpy()
        probs = self.svm.predict_proba(features)[0]
        sorted_indices = np.argsort(probs)[::-1][:3]
        return {
            'label': self.classes[sorted_indices[0]],
            'top3': [(self.classes[i], float(probs[i])) for i in sorted_indices]
        }
