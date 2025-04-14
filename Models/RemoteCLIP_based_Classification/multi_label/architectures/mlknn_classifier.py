
import numpy as np
import torch
from skmultilearn.adapt import MLkNN
from ..core.base import BaseCLIPClassifier

class MLKNNClassifier(BaseCLIPClassifier):
    """多标签K近邻分类器"""
    
    def __init__(self, ckpt_path: str, n_neighbors: int = 10, **kwargs):
        super().__init__(ckpt_path, **kwargs)
        self.n_neighbors = n_neighbors
        self.classifier = MLkNN(k=n_neighbors)

        # 强制禁用混合精度（KNN不需要）  
        self.model.half = lambda: self.model  # 禁用半精度  

    def _train_impl(self, features: np.ndarray, labels: np.ndarray):
        """训练实现"""
        self.classifier.fit(features.astype(np.float32), labels.astype(np.int32))  
        self.logger.info(f"MLKNN trained with {self.n_neighbors} neighbors")

    def _get_predictions(self, data_loader) -> tuple:
        """获取预测结果"""
        all_features, all_labels = [], []
        for images, labels in data_loader:
            features = self._get_features(images.to(self.main_device))  
            all_features.append(features.astype(np.float32))
            all_labels.append(labels.numpy().astype(np.int32))  
        
        return (  
            np.concatenate(all_labels),  
            self.classifier.predict(np.vstack(all_features)).toarray()  
        )  


    def _format_prediction(self, features: np.ndarray) -> dict:
        """格式化预测结果"""
        if isinstance(features, torch.Tensor):  
            features = features.cpu().numpy() 
        
        pred = self.classifier.predict(features.astype(np.float32))  
        return {
            'labels': [int(i) for i, val in enumerate(pred.toarray()[0]) if val == 1]
        }