
import numpy as np
from skmultilearn.adapt import MLkNN
from core.base import BaseCLIPClassifier

class MLKNNClassifier(BaseCLIPClassifier):
    """多标签K近邻分类器"""
    
    def __init__(self, ckpt_path: str, n_neighbors: int = 10, **kwargs):
        super().__init__(ckpt_path, **kwargs)
        self.n_neighbors = n_neighbors
        self.classifier = MLkNN(k=n_neighbors)

    def _train_impl(self, features: np.ndarray, labels: np.ndarray):
        """训练实现"""
        self.classifier.fit(features, labels)
        self.logger.info(f"MLKNN trained with {self.n_neighbors} neighbors")

    def _get_predictions(self, data_loader) -> tuple:
        """获取预测结果"""
        all_features, all_labels = [], []
        for images, labels in data_loader:
            all_features.append(self._get_features(images))
            all_labels.append(labels.numpy())
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        preds = self.classifier.predict(features)
        return labels, preds.toarray()

    def _format_prediction(self, features: np.ndarray) -> dict:
        """格式化预测结果"""
        pred = self.classifier.predict(features)
        return {
            'labels': [int(i) for i, val in enumerate(pred.toarray()[0]) if val == 1]
        }

    def _post_train(self):
        """训练后清理"""
        self.classifier.classifier.clean()  # 清理KNN缓存
