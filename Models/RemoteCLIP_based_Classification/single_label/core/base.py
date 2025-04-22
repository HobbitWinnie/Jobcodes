import os
import logging
from pathlib import Path
from abc import ABC, abstractmethod

import torch
import open_clip
from PIL import Image
import pandas as pd
import numpy as np

class BaseCLIPClassifier(ABC):
    """通用CLIP多模态分类器基类——支持fit类/非fit类子类极简实现"""

    def __init__(
        self, 
        ckpt_path: Path,
        model_name: str = 'ViT-L-14',
        labels=None,
        device_ids: list = None
    ):
        self.labels = labels
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device_ids = device_ids or []
        self._validate_devices()
        self.main_device = self._determine_main_device()
        self.logger.info(f"Main device: {self.main_device}")

        self.clip_model, self.preprocess_func = self._init_clip_model(model_name, ckpt_path)
        self.clip_model = self.clip_model.module if hasattr(self.clip_model, 'module') else self.clip_model
        self.classes = []
        self.label_to_index = {}

    def _init_clip_model(self, model_name, ckpt_path):
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt)
        model = model.to(self.main_device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        model.eval()
        return model, preprocess

    def _validate_devices(self):
        if self.device_ids:
            if not torch.cuda.is_available():
                raise RuntimeError("指定了 device_ids 但 CUDA 不可用")
            available = list(range(torch.cuda.device_count()))
            if any(i not in available for i in self.device_ids):
                raise ValueError(f"无效 device_ids {self.device_ids}, 有效范围: {available}")

    def _determine_main_device(self):
        if self.device_ids:
            return f"cuda:{self.device_ids[0]}"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _get_image_features(self, images):
        images = images.to(self.main_device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled="cuda" in self.main_device):
            features = self.clip_model.encode_image(images)
        return features / features.norm(dim=-1, keepdim=True)

    def _get_text_features(self, texts):
        tokenized = self.tokenizer(texts).to(self.main_device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokenized)
        return features / features.norm(dim=-1, keepdim=True)

    def _extract_features_labels(self, data_loader):
        features, labels = [], []
        for img_batch, label_batch in data_loader:
            feat = self._get_image_features(img_batch).cpu().numpy()
            features.append(feat)
            labels.extend(label_batch)
        features = np.vstack(features)
        return features, labels

    def train(self, train_loader, val_loader=None, **kwargs):
        """
        默认模板：适配sklearn/pytorch式分类器。  
        如果不适合（比如 FewShot/ZeroShot），子类完全可以覆盖本方法。
        """
        features, labels = self._extract_features_labels(train_loader)
        self.classes = sorted(set(labels))
        self.label_to_index = {l: i for i, l in enumerate(self.classes)}
        y = np.array([self.label_to_index[l] for l in labels])
        self._fit_classifier(features, y, **kwargs)
        if val_loader:
            acc = self.evaluate(val_loader)
            self.logger.info(f'Validation accuracy: {acc["accuracy"]:.4f}')

    def evaluate(self, data_loader):
        """
        默认模板：适配sklearn/pytorch式分类器。  
        如果不适合（比如 FewShot/ZeroShot），子类完全可以覆盖本方法。
        """
        correct, total = 0, 0
        for img_batch, label_batch in data_loader:
            features = self._get_image_features(img_batch).cpu().numpy()
            preds = self._predict_batch(features)
            pred_labels = [self.classes[i] for i in preds]
            for pred, gt in zip(pred_labels, label_batch):
                if pred == gt:
                    correct += 1
            total += len(label_batch)
        acc = correct / total if total else 0
        return {'accuracy': acc}

    # 钩子方法：多数常规分类器需实现
    def _fit_classifier(self, features, targets, **kwargs):
        raise NotImplementedError("子类需实现_fit_classifier（如需fit）")

    def _predict_batch(self, features):
        raise NotImplementedError("子类需实现_predict_batch（如需批量推理）")

    # 推理目录/推理单张图片接口，必须实现
    @abstractmethod
    def _predict_single(self, img_path: str) -> dict:
        pass

    def classify_images(self, folder_path: str, output_csv: str):
        results = []
        for img_path in self._iter_images(folder_path):
            try:
                pred = self._predict_single(img_path)
                results.append({"filename": os.path.basename(img_path), **pred})
            except Exception as e:
                self._handle_error(img_path, e)
        self._save_results(results, output_csv)

    def _iter_images(self, folder_path):
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                yield os.path.join(folder_path, fname)

    def _save_results(self, results, output_path):
        pd.DataFrame(results).to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(results)} predictions to {output_path}")

    def _handle_error(self, img_path, error):
        self.logger.error(f"Error processing {img_path}: {str(error)}")

    def _load_image(self, img_path):
        return Image.open(img_path).convert('RGB')
