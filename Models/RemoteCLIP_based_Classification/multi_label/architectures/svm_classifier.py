
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from core.base import BaseCLIPClassifier

class RankSVMClassifier(BaseCLIPClassifier):
    """SVM排序分类器"""
    
    def __init__(self, ckpt_path: str, kernel: str = 'linear', **kwargs):
        super().__init__(ckpt_path, **kwargs)
        self.scaler = StandardScaler()
        self.classifier = OneVsRestClassifier(SVC(kernel=kernel, probability=True))

    def _train_impl(self, features: np.ndarray, labels: np.ndarray):
        """训练实现"""
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
            all_features.append(self._get_features(images))
            all_labels.append(labels.numpy())
        
        features = self.scaler.transform(np.vstack(all_features))
        labels = np.concatenate(all_labels)
        preds = self.classifier.predict(features)
        return labels, preds

    def _format_prediction(self, features: np.ndarray) -> dict:
        """格式化预测结果"""
        scaled_features = self.scaler.transform(features)
        pred = self.classifier.predict(scaled_features)[0]
        proba = self.classifier.predict_proba(scaled_features)[0]
        return {
            'predicted_label': int(pred),
            'probabilities': {str(i): float(p) for i, p in enumerate(proba)}
        }

    def _post_train(self):
        """训练后清理"""
        self.scaler = None  # 释放标准化器内存
