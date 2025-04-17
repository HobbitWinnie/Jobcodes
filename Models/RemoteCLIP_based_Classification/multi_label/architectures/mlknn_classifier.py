
import numpy as np
import torch
from skmultilearn.adapt import MLkNN
from ..core.base import BaseCLIPClassifier

class MLKNNClassifier(BaseCLIPClassifier):  
    """多标签K近邻分类器（适配新版基类）"""  
    
    def __init__(self, ckpt_path: str, num_labels: int, n_neighbors: int = 10, **kwargs):  
        """  
        Args:  
            ckpt_path: CLIP模型权重路径  
            n_neighbors: 近邻数（默认10）  
            kwargs: 传递给基类的参数（如device_ids）  
        """  
        super().__init__(ckpt_path, **kwargs)  
        self.n_neighbors = n_neighbors  
        self.classifier = MLkNN(k=n_neighbors)  
        
        # 强制模型保持FP32精度  
        self.model.float()  
        self.model.half = lambda: self.model  # 禁用半精度  

    def train(self, train_loader, **kwargs):  
        """训练实现（覆盖基类抽象方法）"""  
        # 准备数据（使用基类方法）  
        features, labels = self._prepare_data(train_loader)  
        
        # 类型转换确保稳定性  
        features = features.astype(np.float32)  
        labels = labels.astype(np.int32)  
        
        # 训练分类器  
        self.classifier.fit(features, labels)  
        self.logger.info(f"MLKNN trained with {self.n_neighbors} neighbors")  

    def evaluate(self, data_loader) -> dict:  
        """评估实现（覆盖基类抽象方法）"""  
        y_true, y_pred = self._get_predictions(data_loader)  
        return self._calc_metrics(y_true, y_pred) 
    
    def _get_predictions(self, data_loader) -> tuple:
        """获取预测结果"""
        all_features, all_labels = [], []
        for images, labels in data_loader:
            features = self._get_features(images.to(self.main_device))  
            all_features.append(features.astype(np.float32))
            all_labels.append(labels.numpy().astype(np.int32))  
        
        # 合并数据  
        features = np.vstack(all_features)  
        labels = np.concatenate(all_labels)  
        
        # 返回预测结果  
        return labels, self.classifier.predict(features).toarray()  


    def _format_prediction(self, features: np.ndarray) -> dict:  
        """格式化预测结果（适配新版接口）"""  
        # 处理可能的设备残留  
        if isinstance(features, torch.Tensor):  
            features = features.cpu().numpy()  
            
        # 确保数据类型  
        features = features.astype(np.float32)  
        
        # 获取预测结果  
        pred = self.classifier.predict(features)  
        return {  
            'labels': [int(i) for i, val in enumerate(pred.toarray()[0]) if val == 1],  
            'confidence_scores': pred.toarray()[0].tolist()  
        } 
    
    def __del__(self):  
        """对象销毁时清理资源"""  
        if hasattr(self.classifier, 'classifier'):  
            self.classifier.classifier.clean()  