import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any
from ..core.base import BaseCLIPClassifier

class KNNClassifier(BaseCLIPClassifier):  
    def __init__(self, *args, n_neighbors=20, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)  

    def _fit_classifier(self, features, targets):  
        self.knn.fit(features, targets)  

    def _predict_batch(self, features):  
        return self.knn.predict(features)  

    def _predict_single(self, img_path: str) -> Dict[str, Any]:  
        if not hasattr(self.knn, 'classes_'):  
            raise RuntimeError("KNN尚未训练")  
        image = self._load_image(img_path)  
        img_tensor = self.preprocess_func(image).unsqueeze(0)  
        features = self._get_image_features(img_tensor).cpu().numpy()  
        label_idx = self.knn.predict(features)[0]  
        # top3  
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