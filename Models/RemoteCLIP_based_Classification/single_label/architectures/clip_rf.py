import numpy as np  
from sklearn.ensemble import RandomForestClassifier  
from core.base import BaseCLIPClassifier  

class RF(BaseCLIPClassifier):  
    def __init__(self, *args, n_estimators=100, max_depth=None, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.rf = None  
        self.classes = []  
        self.label_to_index = {}  
        self.n_estimators = n_estimators  
        self.max_depth = max_depth  

    def fit(self, dataloader):  
        """训练随机森林分类器"""  
        features, labels = [], []  
        
        for batch in dataloader:  
            img_batch, label_batch = batch[0], batch[1]  
            feat = self._get_image_features(img_batch).cpu().numpy()  
            features.append(feat)  
            labels.extend(label_batch)  
        
        self.classes = sorted(set(labels))  
        self.label_to_index = {l: i for i, l in enumerate(self.classes)}  
        
        X = np.vstack(features)  
        y = np.array([self.label_to_index[l] for l in labels])  
        
        self.rf = RandomForestClassifier(  
            n_estimators=self.n_estimators,  
            max_depth=self.max_depth  
        )  
        self.rf.fit(X, y)  

    def classify_image(self, image):  
        """实现图像分类"""  
        if not self.rf:  
            raise RuntimeError("Classifier not trained")  
            
        # 预处理和特征提取  
        img_tensor = self.preprocess_func(image).unsqueeze(0)  
        features = self._get_image_features(img_tensor).cpu().numpy()  
        
        # 预测概率  
        probabilities = self.rf.predict_proba(features)[0]  
        sorted_indices = np.argsort(probabilities)[::-1][:3]  
        
        return [  
            (self.classes[i], probabilities[i])   
            for i in sorted_indices  
        ]  