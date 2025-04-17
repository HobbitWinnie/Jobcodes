import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any
from ..core.base import BaseCLIPClassifier

class KNNClassifier(BaseCLIPClassifier):
    def __init__(self, *args, n_neighbors=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors
        self.knn = None
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
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(X, y)

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
        if not self.knn:
            raise RuntimeError("Classifier not trained")
        image = self._load_image(img_path)
        
        img_tensor = self.preprocess_func(image).unsqueeze(0)
        features = self._get_image_features(img_tensor).cpu().numpy()
        label_idx = self.knn.predict(features)[0]
        
        # TopK
        distances, indices = self.knn.kneighbors(features)
        neighbor_labels = [self.classes[self.knn._y[i]] for i in indices[0]]
        prob_dict = {}
        
        for lbl in neighbor_labels:
            prob_dict[lbl] = prob_dict.get(lbl, 0) + 1/len(neighbor_labels)
        top3 = sorted(prob_dict.items(), key=lambda x: -x[1])[:3]
      
        return {
            'label': self.classes[label_idx],
            'top3': [(lbl, float(prob)) for lbl, prob in top3]
        }