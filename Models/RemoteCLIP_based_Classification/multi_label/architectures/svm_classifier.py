import numpy as np  
import torch  
from sklearn.linear_model import SGDClassifier  
from sklearn.multiclass import OneVsRestClassifier  
from sklearn.preprocessing import StandardScaler  
from ..core.base import BaseCLIPClassifier  

class RankSVMClassifier(BaseCLIPClassifier):  
    """多标签SVM（用SGD拟合，适配新版基类）"""  

    def __init__(self, ckpt_path: str, num_labels: int, **kwargs):  
        super().__init__(ckpt_path, **kwargs)  
        self.num_labels = num_labels  
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
        self.clip_model = self.clip_model.module if hasattr(self.clip_model, 'module') else self.clip_model  


    def train(self, train_loader, val_loader=None, **kwargs):  

        features, labels = self._prepare_features_labels(train_loader)  
        self.logger.info("Extracting features for RankSVM training...")  
        assert features.ndim == 2, "features shape 必须为 (N, D)"  
        assert labels.ndim == 2 and labels.shape[1] > 1, "labels 必须为 (N, K) 多标签格式"  
        features = features.astype(np.float32)  
        labels = labels.astype(np.int32)  

        # 标准化  
        self.scaler.fit(features)  
        scaled_features = self.scaler.transform(features)  

        # 训练SVM  
        self.classifier.fit(scaled_features, labels)  
        self.logger.info("RankSVM(SGD) 训练完毕")  

        if val_loader is not None:  
            metrics = self.evaluate(val_loader)  
            self.logger.info(f"Val F1: {metrics[0]:.4f}, F2: {metrics[1]:.4f}")  

    def evaluate(self, data_loader) -> dict:  
        y_true, y_pred = self._get_predictions(data_loader)  
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
        scaled_features = self.scaler.transform(features)  
        preds = self.classifier.predict(scaled_features)  
        return labels, preds  

    def _format_prediction(self, image) -> dict:  
        # 单张图片推理  
        img_tensor = self.preprocess_func(image).unsqueeze(0).to(self.main_device)  
        with torch.no_grad():  
            feats = self.clip_model.encode_image(img_tensor)  
            feats = feats / feats.norm(dim=-1, keepdim=True)  
        feats_np = feats.cpu().numpy().astype(np.float32)  
        scaled_feats = self.scaler.transform(feats_np)  
        pred = self.classifier.predict(scaled_feats)[0]  
        return {  
            "labels": [i for i, val in enumerate(pred) if val == 1],  
            "confidence_scores": pred.tolist()  
        }