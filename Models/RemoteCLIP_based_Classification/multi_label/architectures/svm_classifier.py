import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from ..core.base import BaseCLIPClassifier

class RankSVMClassifier(BaseCLIPClassifier):
    """多标签 SVM (SGD近似) 分类器"""

    def __init__(self, ckpt_path: str, num_labels: int, **kwargs):
        super().__init__(ckpt_path, **kwargs)
        self.scaler = StandardScaler()
        self.classifier = OneVsRestClassifier(
            SGDClassifier(
                loss='hinge',   
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                tol=1e-3,
                n_jobs=2,
                random_state=42
            ),
            n_jobs=2
        )
        self.model.float()
        self.model.half = lambda: self.model  # 禁用半精度

    def train(self, train_loader, val_loader=None, **kwargs):
        features, labels = self._prepare_data(train_loader)

        # 检查和处理标签与特征
        assert features.ndim == 2, "features shape 必须为 (N, D)"
        assert labels.ndim == 2 and labels.shape[1] > 1, "labels 必须为 (N, K) 多标签格式"
        print("训练集shape:", features.shape, labels.shape)

        features = features.astype(np.float32)
        labels = labels.astype(np.int32)

        # 标准化
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)

        # 训练分类器
        self.classifier.fit(scaled_features, labels)
        self.logger.info("RankSVM(SGD) 训练完毕")

        # 验证逻辑  
        if val_loader:  
            metrics = self.evaluate(val_loader)
            self.logger.info(f"Test F1: {metrics['f1']:.4f}") 


    def evaluate(self, data_loader) -> dict:
        y_true, y_pred = self._get_predictions(data_loader)
        return self._calc_metrics(y_true, y_pred)

    def _get_predictions(self, data_loader) -> tuple:
        all_features, all_labels = [], []
        for images, labels in data_loader:
            features = self._get_features(images.to(self.main_device))
            all_features.append(features.astype(np.float32))
            all_labels.append(labels.numpy().astype(np.int32))

        features = np.vstack(all_features)
        labels = np.vstack(all_labels)
        scaled_features = self.scaler.transform(features)
        preds = self.classifier.predict(scaled_features)
        return labels, preds

    def _format_prediction(self, features: np.ndarray) -> dict:
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        features = features.astype(np.float32)
        scaled_features = self.scaler.transform(features)
        pred = self.classifier.predict(scaled_features)
        return {
            'labels': [int(i) for i, val in enumerate(pred[0]) if val == 1],
            'confidence_scores': pred[0].tolist()
        }