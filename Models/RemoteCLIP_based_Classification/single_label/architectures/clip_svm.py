import numpy as np
from sklearn.svm import SVC
from typing import Dict, Any
from ..core.base import BaseCLIPClassifier

class SVMClassifier(BaseCLIPClassifier):  
    def __init__(self, *args, C=1.0, kernel='linear', **kwargs):  
        super().__init__(*args, **kwargs)  
        self.svm = SVC(C=C, kernel=kernel, probability=True)  

    def _fit_classifier(self, features, targets):  
        self.svm.fit(features, targets)  

    def _predict_batch(self, features):  
        return self.svm.predict(features)  

    def _predict_single(self, img_path: str) -> Dict[str, Any]:  
        if self.svm is None:  
            raise RuntimeError("SVM尚未训练")  
        image = self._load_image(img_path)  
        img_tensor = self.preprocess_func(image).unsqueeze(0)  
        features = self._get_image_features(img_tensor).cpu().numpy()  
        probs = self.svm.predict_proba(features)[0]  
        sorted_indices = np.argsort(probs)[::-1][:3]  
        return {  
            'label': self.classes[sorted_indices[0]],  
            'top3': [(self.classes[i], float(probs[i])) for i in sorted_indices]  
        }  