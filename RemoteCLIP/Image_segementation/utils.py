import os
import logging
from typing import Tuple, Dict, List, Union
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(log_dir: str = None):
    """设置日志配置"""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

class CombinedLoss(nn.Module):
    """组合损失函数：CrossEntropy + Dice Loss"""
    
    def __init__(self, weights: List[float] = [0.5, 0.5], ignore_index: int = 0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.weights = weights
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(pred, target)
        
        pred_soft = F.softmax(pred, dim=1)
        dice_loss = 1 - dice_coefficient(pred_soft, target, self.ignore_index)
        
        return self.weights[0] * ce_loss + self.weights[1] * dice_loss

def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 0,
    smooth: float = 1e-8
) -> torch.Tensor:
    """计算Dice系数"""
    num_classes = pred.size(1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2)
    
    dice_scores = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        pred_cls = pred[:, cls]
        target_cls = target_one_hot[:, cls].float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
        
    return torch.mean(torch.stack(dice_scores))

def load_and_save_data(image_path, label_path, output_dir, normalize = True):  
    """  
    加载并处理遥感图像和标签数据，支持数据预处理和可选的保存功能  
    
    Args:  
        image_path: 图像文件路径  
        label_path: 标签文件路径（可选）  
        output_dir: 输出目录（可选），如果不指定则不保存处理后的数据  
        normalize: 是否进行归一化  
        
    Returns:  
        处理后的图像数据、标签数据和元数据  
    """  
    # 如果指定了输出目录，则创建相应的目录结构  
    if output_dir is not None:  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        output_dir = os.path.join(output_dir, timestamp)  
        os.makedirs(output_dir, exist_ok=True)  
    
    # 加载并处理图像数据  
    try:  
        logging.info(f"Loading image from {image_path}")  
        with rasterio.open(image_path) as src:  
            image = src.read()  
            image_nodata = int(src.nodata)  
            image_meta = src.meta  
            
        # 处理无效值  
        image_mask = np.ones_like(image[0], dtype=bool)  
        if image_nodata is not None:  
            image_mask = (image[0] != image_nodata)  
            
        # 数据有效性检查  
        if np.all(~image_mask):  
            raise ValueError("No valid data in image")  
            
        # 替换无效值为0  
        image = np.where(np.broadcast_to(image_mask, image.shape), image, 0)  
        
        # 数据归一化  
        if normalize:  
            image_normalized = np.zeros_like(image, dtype=np.float32)  
            for i in range(image.shape[0]):  
                valid_data = image[i][image_mask]  
                if len(valid_data) > 0:  
                    min_val = np.percentile(valid_data, 1)  
                    max_val = np.percentile(valid_data, 99)  
                    image_normalized[i] = np.clip((image[i] - min_val) / (max_val - min_val + 1e-8), 0, 1)  
            image = image_normalized  
            
        # 如果指定了输出目录，则保存处理后的图像  
        if output_dir is not None:  
            processed_image_path = os.path.join(output_dir, 'processed_image.tif')  
            with rasterio.open(  
                processed_image_path,  
                'w',  
                driver='GTiff',  
                height=image.shape[1],  
                width=image.shape[2],  
                count=image.shape[0],  
                dtype=image.dtype,  
                crs=image_meta['crs'],  
                transform=image_meta['transform']  
            ) as dst:  
                dst.write(image)  
                if normalize:  
                    dst.write_mask(image_mask.astype(np.uint8))  
        
    except Exception as e:  
        logging.error(f"Error processing image: {e}")  
        raise  
        
    # 加载并处理标签数据  
    labels = None  
    if label_path:  
        try:  
            logging.info(f"Loading labels from {label_path}")  
            with rasterio.open(label_path) as src:  
                labels = src.read(1)  
                labels_nodata = int(src.nodata)  
                
            # 处理标签无效值  
            label_mask = np.ones_like(labels, dtype=bool)  
            if labels_nodata is not None:  
                label_mask = (labels != labels_nodata)  
                
            # 替换无效值为0  
            labels = np.where(label_mask, labels, 0)  
            
            # 如果指定了输出目录，则保存处理后的标签  
            if output_dir is not None:  
                processed_label_path = os.path.join(output_dir, 'processed_labels.tif')  
                with rasterio.open(  
                    processed_label_path,  
                    'w',  
                    driver='GTiff',  
                    height=labels.shape[0],  
                    width=labels.shape[1],  
                    count=1,  
                    dtype=labels.dtype,  
                    crs=image_meta['crs'],  
                    transform=image_meta['transform']  
                ) as dst:  
                    dst.write(labels, 1)  
                    dst.write_mask(label_mask.astype(np.uint8))  
                
        except Exception as e:  
            logging.error(f"Error processing labels: {e}")  
            raise  
    
    # 记录处理信息  
    logging.info(f"Image shape: {image.shape}, dtype: {image.dtype}")  
    if labels is not None:  
        logging.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")  
    
    if output_dir is not None:  
        logging.info(f"Data processing completed. Results saved to {output_dir}")  
        
    return image, labels, image_meta  


def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0
) -> Dict[str, float]:
    """计算评估指标"""
    pred_cls = pred.argmax(dim=1)
    
    # 计算总体准确率
    valid_pixels = target != ignore_index
    accuracy = (pred_cls[valid_pixels] == target[valid_pixels]).float().mean()
    
    # 计算IoU
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        
        pred_mask = pred_cls == cls
        target_mask = target == cls
        
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        
        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou.item())
    
    mean_iou = np.mean(ious)
    
    # 计算Dice系数
    dice = dice_coefficient(F.softmax(pred, dim=1), target, ignore_index)
    
    return {
        'accuracy': accuracy.item(),
        'mean_iou': mean_iou,
        'dice': dice.item()
    }

def set_random_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
