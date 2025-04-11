import os  
import logging  
from pathlib import Path  
from typing import List, Tuple, Dict, Optional, Any  
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
            device: Optional[torch.device] = None
        ):  
        """  
        初始化分类器  
        
        :param ckpt_path: 模型检查点路径  
        :param model_name: CLIP模型名称 (默认: ViT-L-14)  
        :param device: 指定运行设备 (默认自动选择)  
        """  
        # 参数校验  
        if not ckpt_path.exists():  
            raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")  

        self.model_name = model_name  
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  
        
        self._init_model(ckpt_path)  

    def _init_model(self, ckpt_path: Path) -> None:  
        """初始化CLIP模型组件"""  
        try:  
            # 加载模型组件  
            self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(  
                self.model_name  
            )  
            self.tokenizer = open_clip.get_tokenizer(self.model_name)  
            
            # 加载检查点  
            ckpt = torch.load(ckpt_path, map_location="cpu")  
            self.model.load_state_dict(ckpt)  
            
            # 设备转移  
            self.model = self.model.to(self.device).eval()  
            logger.info(f"模型加载完成，运行设备: {self.device}")  
            
        except Exception as e:  
            logger.error(f"模型初始化失败: {str(e)}")  
            raise  


    def _get_image_features(self, images):  
        """  
        提取图像特征  
        
        :param images: 预处理后的图像张量 (形状: [B, C, H, W])  
        :return: 归一化后的特征向量 (形状: [B, D])  
        """  
        try:  
            images = images.to(self.device)  
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):  
                features = self.model.encode_image(images)  
            return features / features.norm(dim=-1, keepdim=True)  
            
        except RuntimeError as e:  
            logger.error(f"特征提取失败: {str(e)}")  
            raise  

    def _get_text_features(self, texts):  
        """  
        提取文本特征  
        
        :param texts: 输入文本列表  
        :return: 归一化后的特征向量 (形状: [N, D])  
        """  
        try:  
            tokenized = self.tokenizer(texts).to(self.device)  
            with torch.no_grad():  
                features = self.model.encode_text(tokenized)  
            return features / features.norm(dim=-1, keepdim=True)  
            
        except Exception as e:  
            logger.error(f"文本处理失败: {str(e)}")  
            raise  

    @abstractmethod  
    def classify_image(self, image):  
        """抽象分类方法"""  
        pass  

    def classify_images_in_folder(  
        self,  
        folder_path: Path,  
        output_csv: Path,  
        max_retries: int = 3  
    ) -> None:  
        """  
        批量分类文件夹中的图像  
        
        :param folder_path: 输入文件夹路径  
        :param output_csv: 输出CSV路径  
        :param max_retries: 单文件最大重试次数  
        """  
        if not folder_path.is_dir():  
            raise NotADirectoryError(f"无效文件夹路径: {folder_path}")  

        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = folder_path / img_name  
            
            # 跳过非图像文件  
            if not img_path.suffix.lower()[1:] in {"png", "jpg", "jpeg", "bmp", "gif"}:  
                continue  
                
            # 带重试机制的图像加载  
            for attempt in range(max_retries):  
                try:  
                    image = Image.open(img_path).convert("RGB")  
                    preds = self.classify_image(image)  
                    results.append({
                        "filename": img_name,
                        **{f"top{i+1}_label": l[0]: l[1] for i, l in enumerate(preds)}
                    })
                    break  
                except (OSError, Image.UnidentifiedImageError) as e:  
                    logger.warning(f"文件损坏 [{img_name}] 第{attempt+1}次重试: {str(e)}")  
                    if attempt == max_retries - 1:  
                        logger.error(f"无法处理文件: {img_name}")  
                        break  
                except Exception as e:  
                    logger.error(f"处理失败 [{img_name}]: {str(e)}")  
                    break  

        # 保存结果  
        if results:  
            pd.DataFrame(results).to_csv(output_csv, index=False)  
            logger.info(f"结果已保存至: {output_csv}")  
        else:  
            logger.warning("未处理任何有效图像")  
