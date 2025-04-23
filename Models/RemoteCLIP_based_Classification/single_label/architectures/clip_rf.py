import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ..core.base import BaseCLIPClassifier

class RFClassifier(BaseCLIPClassifier):  
    def __init__(self, *args, n_estimators=100, max_depth=None, random_state=42, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.rf = RandomForestClassifier(  
            n_estimators=n_estimators,  
            max_depth=max_depth,  
            random_state=random_state  
        )  

    def _fit_classifier(self, features, targets):  
        self.logger.info(f'{self.__class__.__name__} classifier training begin')
        self.rf.fit(features, targets)  
        self.logger.info(f'{self.__class__.__name__} classifier training completed')

    def _predict_batch(self, features):  
        return self.rf.predict(features)  

    def _predict_single(self, img_path: str):  
        if self.rf is None:  
            raise RuntimeError("RandomForest尚未训练")  
        image = self._load_image(img_path)  
        img_tensor = self.preprocess_func(image).unsqueeze(0)  
        features = self._get_image_features(img_tensor).cpu().numpy()  
        probs = self.rf.predict_proba(features)[0]  
        sorted_indices = np.argsort(probs)[::-1][:3]  
        return {  
            'label': self.classes[sorted_indices[0]],  
            'top3': [(self.classes[i], float(probs[i])) for i in sorted_indices]  
        }  