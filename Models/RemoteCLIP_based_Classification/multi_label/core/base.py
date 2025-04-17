
import os
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from PIL import Image
from sklearn.metrics import f1_score, fbeta_score
import torch
import open_clip

class BaseCLIPClassifier(ABC):
    """CLIP分类器基类（模板方法模式）"""
    
    def __init__(self, ckpt_path: str,   
                 model_name: str = 'ViT-L-14',  
                 device_ids: list = None):  
        """  
        Args:  
            ckpt_path: CLIP模型权重路径  
            model_name: CLIP模型名称  
            device_ids: 使用的GPU设备ID列表（空列表时自动选择可用设备）  
        """ 
        # 初始化日志  
        self.logger = logging.getLogger(self.__class__.__name__)  
        
        # 初始化设备配置  
        self.device_ids = device_ids or []  
        self._validate_devices()  
        self.main_device = self._determine_main_device()        
        self.logger.info(f"Main device: {self.main_device}")  

        # 模型初始化  
        self.model, self.preprocess_func = self._init_clip_model(model_name, ckpt_path)  
        self.logger.info(f"Loaded {model_name} from {ckpt_path}")  


    def _init_clip_model(self, model_name: str, ckpt_path: str) -> tuple:  
        """初始化CLIP模型核心方法"""  
        # 创建模型  
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)  
        
        # 加载权重  
        ckpt = torch.load(ckpt_path, map_location='cpu')  
        model.load_state_dict(ckpt)  
        
        # 设备配置  
        model = model.to(self.main_device)  
        if len(self.device_ids) > 1:  
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)  
        model.eval()  
        
        return model, preprocess 


    def _validate_devices(self):  
        """设备配置验证"""  
        if self.device_ids:  
            if not torch.cuda.is_available():  
                raise RuntimeError("CUDA不可用时不能指定device_ids")  
                
            available_ids = list(range(torch.cuda.device_count()))  
            if any(idx not in available_ids for idx in self.device_ids):  
                raise ValueError(  
                    f"无效的device_ids {self.device_ids}，可用设备: {available_ids}"  
                )  

    def _determine_main_device(self) -> str:  
        """自动确定主计算设备"""  
        if self.device_ids:  
            return f"cuda:{self.device_ids[0]}"  
        return "cuda:0" if torch.cuda.is_available() else "cpu"  
    
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
    
    def _extract_feature_label_tensors(self, data_loader):
        """特征及标签整合为Tensor，用于FC训练"""
        features, labels = [], []
        self.model.eval()  # 主干进入评估模式
        for batch_imgs, batch_labels in data_loader:
            batch_imgs = batch_imgs.to(self.main_device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.main_device.startswith('cuda')):
                # torch.nn.DataParallel: model自动均分到多卡
                model = self.model.module if hasattr(self.model, 'module') else self.model
                feats = model.encode_image(batch_imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu())
            labels.append(batch_labels.cpu())
        features = torch.cat(features, dim=0)  # [N, D]
        labels = torch.cat(labels, dim=0)      # [N, C]
        return features, labels
    
    def _get_features(self, images):
        """获取图像特征"""
        images = images.to(self.main_device)  

        # 特征提取  
        with torch.no_grad(), torch.cuda.amp.autocast(enabled='cuda' in self.main_device):  
            # 处理DataParallel封装  
            model = self.model.module if hasattr(self.model, 'module') else self.model  
            features = model.encode_image(images)  # 从实际模型获取特征  
        
        # 特征归一化处理  
        features = features / features.norm(dim=-1, keepdim=True)  
        return features.cpu().numpy().astype(np.float32)  # 统一返回float32  
    
    def _calc_metrics(self, y_true, y_pred):
        """计算评估指标"""
        threshold = 0.5  
        y_pred_bin = (y_pred > threshold).astype(int)  
        return {
            'f1': f1_score(y_true, y_pred_bin, average='macro', zero_division=1),
            'f2': fbeta_score(y_true, y_pred_bin, beta=2, average='macro', zero_division=1)
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

    def _save_results(self, results, output_path):
        """保存结果"""
        pd.DataFrame(results).to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(results)} predictions to {output_path}")

    def _handle_error(self, img_path, error):
        """错误处理（可重写）"""
        self.logger.error(f"Error processing {img_path}: {str(error)}")

        
    @abstractmethod
    def _format_prediction(self, features: np.ndarray) -> dict:  
        """预测结果格式化方法（子类必须实现）"""  
        pass  

    @abstractmethod  
    def train(self, train_loader, **kwargs):  
        """训练入口（子类必须实现）"""  
        pass  

    @abstractmethod  
    def evaluate(self, data_loader) -> dict:  
        """评估入口（子类必须实现）"""  
        pass  
