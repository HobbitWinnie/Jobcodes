import numpy as np  
from sklearn.svm import SVC  
from core.base import BaseCLIPClassifier  

class SVM(BaseCLIPClassifier):  
    def __init__(self, *args, C=1.0, kernel='linear', **kwargs):  
        super().__init__(*args, **kwargs)  
        self.svm = None  
        self.classes = []  
        self.label_to_index = {}  
        self.C = C  
        self.kernel = kernel  

    def fit(self, dataloader):  
        """训练SVM分类器"""  
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
        
        self.svm = SVC(  
            C=self.C,   
            kernel=self.kernel,  
            probability=True  
        )  
        self.svm.fit(X, y)  

    def classify_image(self, image):  
        """实现图像分类"""  
        if not self.svm:  
            raise RuntimeError("Classifier not trained")  
            
        # 预处理和特征提取  
        img_tensor = self.preprocess_func(image).unsqueeze(0)  
        features = self._get_image_features(img_tensor).cpu().numpy()  
        
        # 预测概率  
        probabilities = self.svm.predict_proba(features)[0]  
        sorted_indices = np.argsort(probabilities)[::-1][:3]  
        
        return [  
            (self.classes[i], probabilities[i])   
            for i in sorted_indices  
        ]  