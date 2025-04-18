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
            num_labels: 标签数  
            n_neighbors: 近邻数（默认10）  
            kwargs: 传递给基类的参数（如device_ids等）  
        """  
        super().__init__(ckpt_path, **kwargs)  
        self.n_neighbors = n_neighbors  
        self.num_labels = num_labels  
        self.classifier = MLkNN(k=n_neighbors)  # 仅CPU实现  
        self.clip_model = self.clip_model.module if hasattr(self.clip_model, 'module') else self.clip_model  

    def train(self, train_loader, val_loader=None, **kwargs):  
        """提取全部特征并训练MLKNN"""  
        self.logger.info("Extracting features for MLKNN training...")  
        features, labels = self._prepare_features_labels(train_loader)  
        features = features.astype(np.float32)  
        labels = labels.astype(np.int32)  
        self.classifier.fit(features, labels)  
        self.logger.info(f"MLKNN trained with {self.n_neighbors} neighbors")  

        if val_loader is not None:  
            metrics = self.evaluate(val_loader)  
            self.logger.info(f"Val metrics: F1 = {metrics[0]:.4f}, F2 = {metrics[1]:.4f}")  

    def evaluate(self, data_loader) -> dict:  
        """提取特征后评价MLKNN"""  
        y_true, y_pred = self._get_predictions(data_loader)  
        # y_true, y_pred 都是numpy二维数组  
        return self._calc_metrics(y_true, y_pred)  

    def _prepare_features_labels(self, loader):  
        features_list, labels_list = [], []  
        with torch.no_grad():  
            for images, labels in loader:  
                images = images.to(self.main_device)  
                feats = self.clip_model.encode_image(images)  
                feats = feats / feats.norm(dim=-1, keepdim=True)  
                feats = feats.cpu().numpy()  
                features_list.append(feats.astype(np.float32))  
                labels_list.append(labels.cpu().numpy().astype(np.int32))  
        features = np.vstack(features_list)  
        labels = np.vstack(labels_list)  
        return features, labels  

    def _get_predictions(self, data_loader):  
        features, labels = self._prepare_features_labels(data_loader)  
        preds = self.classifier.predict(features).toarray()           # shape: (n_samples, n_labels)  
        return labels, preds  

    def _format_prediction(self, image) -> dict:  
        """兼容predict_single调用/图片输入的预测"""  
        # 1. 预处理图片  
        img_tensor = self.preprocess_func(image).unsqueeze(0).to(self.main_device)  
        with torch.no_grad():  
            feats = self.clip_model.encode_image(img_tensor)  
            feats = feats / feats.norm(dim=-1, keepdim=True)  
        feats_np = feats.cpu().numpy().astype(np.float32)  
        # 2. MLKNN推理  
        pred = self.classifier.predict(feats_np).toarray()[0]  
        labels = [i for i, val in enumerate(pred) if val]  
        return {  
            "labels": labels,  
            "confidence_scores": pred.tolist()  
        }  