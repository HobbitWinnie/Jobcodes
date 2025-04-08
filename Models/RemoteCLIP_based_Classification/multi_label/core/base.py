
import os
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from PIL import Image
from sklearn.metrics import f1_score, fbeta_score

class BaseCLIPClassifier(ABC):
    """CLIP分类器基类（模板方法模式）"""
    
    def __init__(self, ckpt_path: str, model_name: str = 'ViT-L-14', device: str = None):
        # 初始化CLIP模型
        import torch
        import open_clip
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(model_name)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device).eval()
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self, train_loader, val_loader=None, **kwargs):
        """训练模板方法"""
        self.logger.info("Training started")
        try:
            # 通用预处理
            features, labels = self._prepare_data(train_loader)
            
            # 子类具体训练
            self._train_impl(features, labels, **kwargs)
            
            # 验证流程
            if val_loader:
                self.logger.info("Running validation")
                results = self.evaluate(val_loader)
                self.logger.info(f"Validation scores: {results}")
                
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            self._post_train()

    def evaluate(self, data_loader) -> dict:
        """评估模板方法"""
        self.logger.info("Evaluation started")
        y_true, y_pred = self._get_predictions(data_loader)
        return self._calc_metrics(y_true, y_pred)

    def classify_images(self, folder_path: str, output_csv: str):
        """批量分类模板方法"""
        results = []
        for img_path in self._iter_images(folder_path):
            try:
                pred = self._predict_single(img_path)
                results.append({"filename": os.path.basename(img_path), **pred})
            except Exception as e:
                self._handle_error(img_path, e)
        self._save_results(results, output_csv)

    @abstractmethod
    def _train_impl(self, features: np.ndarray, labels: np.ndarray, **kwargs): ...

    @abstractmethod
    def _get_predictions(self, data_loader) -> tuple: ...

    # 以下为通用实现
    def _prepare_data(self, loader):
        """准备训练数据"""
        features, labels = [], []
        for images, batch_labels in loader:
            features.append(self._get_features(images))
            labels.append(batch_labels.numpy())
        return np.vstack(features), np.concatenate(labels)

    def _get_features(self, images):
        """获取图像特征"""
        import torch
        images = images.to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled='cuda' in self.device):
            features = self.model.encode_image(images)
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

    def _calc_metrics(self, y_true, y_pred):
        """计算评估指标"""
        return {
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=1),
            'f2': fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=1)
        }

    def _iter_images(self, folder_path):
        """迭代有效图像文件"""
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                yield os.path.join(folder_path, fname)

    def _predict_single(self, img_path):
        """单图预测流程"""
        image = Image.open(img_path).convert('RGB')
        tensor = self.preprocess_func(image).unsqueeze(0)
        features = self._get_features(tensor)
        return self._format_prediction(features)

    @abstractmethod
    def _format_prediction(self, features): ...

    def _save_results(self, results, output_path):
        """保存结果"""
        pd.DataFrame(results).to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(results)} predictions to {output_path}")

    def _post_train(self):
        """训练后清理（可重写）"""
        pass

    def _handle_error(self, img_path, error):
        """错误处理（可重写）"""
        self.logger.error(f"Error processing {img_path}: {str(error)}")
