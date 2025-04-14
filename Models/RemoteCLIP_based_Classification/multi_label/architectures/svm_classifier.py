
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from ..core.base import BaseCLIPClassifier

class RankSVMClassifier(BaseCLIPClassifier):
    """SVM排序分类器"""
    
    def __init__(self, ckpt_path: str, kernel: str = 'linear', **kwargs):
        super().__init__(ckpt_path, **kwargs)
        self.scaler = StandardScaler()
        self.classifier = OneVsRestClassifier(SVC(kernel=kernel, probability=True))

        # 禁用混合精度并锁定模型精度  
        self.model.float()  # 强制使用FP32  
        self.model.half = lambda: self.model  # 防止意外转为半精度  

    def _train_impl(self, features: np.ndarray, labels: np.ndarray):
        """训练实现"""
        # 确保使用CPU numpy数组  
        features = features.astype(np.float32)  
        labels = labels.astype(np.int32) 

        # 特征标准化
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        # 训练分类器
        self.classifier.fit(scaled_features, labels)
        self.logger.info(f"SVM trained with {self.classifier.estimator.kernel} kernel")

    def _get_predictions(self, data_loader) -> tuple:
        """获取预测结果"""
        all_features, all_labels = [], []
        for images, labels in data_loader:
            features = self._get_features(images.to(self.main_device))  
            all_features.append(features.astype(np.float32))  
            all_labels.append(labels.numpy().astype(np.int32))  
        
       # 标准化处理  
        scaled_features = self.scaler.transform(np.vstack(all_features))  
        return np.concatenate(all_labels), self.classifier.predict(scaled_features)  

    def _format_prediction(self, features: np.ndarray) -> dict:
        """格式化预测结果"""
        # 处理可能的设备残留  
        if isinstance(features, torch.Tensor):  
            features = features.cpu().numpy().astype(np.float32)  

        scaled_features = self.scaler.transform(features)  
        pred = self.classifier.predict(scaled_features)[0]  
        proba = self.classifier.predict_proba(scaled_features)[0]  
        return {  
            'predicted_label': int(pred),  
            'probabilities': {str(i): float(p) for i, p in enumerate(proba)}  
        } 
    
    def __del__(self):  
        """对象销毁时清理SVM资源"""  
        if hasattr(self.classifier, 'estimators_'):  
            del self.classifier.estimators_  