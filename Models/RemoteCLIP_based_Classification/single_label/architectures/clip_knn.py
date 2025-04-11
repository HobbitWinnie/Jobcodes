import numpy as np  
from sklearn.neighbors import KNeighborsClassifier  
from nw.Codes.Models.RemoteCLIP_based_Classification.single_label.core.base import BaseCLIPClassifier  

class KNN(BaseCLIPClassifier):  
    def __init__(self, *args, n_neighbors=20, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.knn = None  
        self.classes = []  
        self.label_to_index = {}  

    def fit(self, dataloader):  
        """训练KNN分类器"""  
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
        
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)  
        self.knn.fit(X, y)  

    def classify_image(self, image):  
        """实现图像分类"""  
        if not self.knn:  
            raise RuntimeError("Classifier not trained")  
            
        # 预处理和特征提取  
        img_tensor = self.preprocess_func(image).unsqueeze(0)  
        features = self._get_image_features(img_tensor).cpu().numpy()  
        
        # KNN预测  
        distances, indices = self.knn.kneighbors(features)  
        neighbor_labels = [self.classes[self.knn._y[i]] for i in indices[0]]  
        
        # 计算概率  
        prob_dict = {}  
        for lbl in neighbor_labels:  
            prob_dict[lbl] = prob_dict.get(lbl, 0) + 1/len(neighbor_labels)  
            
        return sorted(prob_dict.items(), key=lambda x: -x[1])[:3]  