import os  
import logging  
from pathlib import Path  
from typing import List, Dict, Any  
import torch  
import open_clip  
from PIL import Image  
import pandas as pd  
from abc import ABC, abstractmethod  


class BaseCLIPClassifier(ABC):  
    """CLIP分类器基类 (支持多模态分类方法)"""  

    def __init__(
            self, 
            ckpt_path: Path,
            model_name: str ='ViT-L-14', 
            device_ids: list = None
        ):  
        
        # 初始化日志  
        self.logger = logging.getLogger(self.__class__.__name__)  

        # 初始化设备配置  
        self.device_ids = device_ids or []  
        self._validate_devices()  
        self.main_device = self._determine_main_device()        
        self.logger.info(f"Main device: {self.main_device}")  

        # 模型初始化  
        self.clip_model, self.preprocess_func = self._init_clip_model(model_name, ckpt_path)  
        self.logger.info(f"Loaded {model_name} from {ckpt_path}")  
        
        # DataParallel兼容
        self.clip_model = self.clip_model.module if hasattr(self.clip_model, 'module') else self.clip_model  

    def _init_clip_model(self, model_name: str, ckpt_path: str) -> tuple:  
        """初始化CLIP模型核心方法"""  
        # 创建模型  
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)  
        self.tokenizer = open_clip.get_tokenizer(model_name)  

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
    
    def _get_image_features(self, images):  
        """  
        提取图像特征  
        
        :param images: 预处理后的图像张量 (形状: [B, C, H, W])  
        :return: 归一化后的特征向量 (形状: [B, D])  
        """  
        try:  
            images = images.to(self.main_device)  
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=("cuda" in self.main_device)):  
                features = self.clip_model.encode_image(images)  
            return features / features.norm(dim=-1, keepdim=True)  
            
        except RuntimeError as e:  
            self.logger.error(f"特征提取失败: {str(e)}")  
            raise  

    def _get_text_features(self, texts):  
        """  
        提取文本特征  
        
        :param texts: 输入文本列表  
        :return: 归一化后的特征向量 (形状: [N, D])  
        """  
        try:  
            tokenized = self.tokenizer(texts).to(self.main_device)  
            with torch.no_grad():  
                features = self.clip_model.encode_text(tokenized)  
            return features / features.norm(dim=-1, keepdim=True)  
            
        except Exception as e:  
            self.logger.error(f"文本处理失败: {str(e)}")  
            raise  

    @abstractmethod  
    def _predict_single(self, img_path: str):  
        """抽象分类方法"""  
        pass  
    
    @abstractmethod  
    def train(self, train_loader, **kwargs):  
        """训练入口（子类必须实现）"""  
        pass  

    @abstractmethod  
    def evaluate(self, data_loader) -> dict:  
        """评估入口（子类必须实现）"""  
        pass  

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

    def _iter_images(self, folder_path):
        """迭代有效图像文件"""
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                yield os.path.join(folder_path, fname)
    
    def _save_results(self, results, output_path):
        """保存结果"""
        pd.DataFrame(results).to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(results)} predictions to {output_path}")

    def _handle_error(self, img_path, error):
        """错误处理（可重写）"""
        self.logger.error(f"Error processing {img_path}: {str(error)}")

    def _load_image(self, img_path):  
        return Image.open(img_path).convert('RGB')  