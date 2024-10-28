import os
import logging
from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import rasterio  


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EarlyStopping:  
    def __init__(self, patience=100, min_delta=1e-4):  
        self.patience = patience  
        self.min_delta = min_delta  
        self.counter = 0  
        self.best_loss = None  
        self.should_stop = False  

    def __call__(self, val_loss):  
        if self.best_loss is None:  
            self.best_loss = val_loss  
        elif val_loss > self.best_loss - self.min_delta:  
            self.counter += 1  
            if self.counter >= self.patience:  
                self.should_stop = True  
        else:  
            self.best_loss = val_loss  
            self.counter = 0  

class CombinedLoss(nn.Module):  
    def __init__(self, weights=[0.5, 0.5], ignore_index=0):  
        """  
        组合损失函数：CrossEntropy + Dice Loss  
        
        Args:  
            weights: 两种损失的权重 [ce_weight, dice_weight]  
            ignore_index: 忽略的类别索引，默认为0（nodata）  
        """  
        super().__init__()  
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')  
        self.weights = weights  
        self.ignore_index = ignore_index  
        
    def forward(self, outputs, targets):  
        # 创建mask排除nodata区域  
        mask = (targets != self.ignore_index).float()  
        
        # 计算CrossEntropy损失  
        ce_loss = self.ce(outputs, targets)  
        masked_ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)  
        
        # 计算Dice损失，排除nodata区域  
        dice_loss = 1 - multiclass_dice_coefficient(  
            F.softmax(outputs, dim=1),  
            targets,  
            mask,  
            ignore_index=self.ignore_index  
        )  
        
        # 组合损失  
        total_loss = self.weights[0] * masked_ce_loss + self.weights[1] * dice_loss  
        
        return total_loss, {  
            'ce_loss': masked_ce_loss.item(),  
            'dice_loss': dice_loss.item()  
        }  
  
def multiclass_dice_coefficient(predictions, targets, mask, ignore_index=0, smooth=1e-8):  
    """  
    计算多类别的Dice系数，排除nodata区域  
    
    Args:  
        predictions: 模型预测结果 (B, C, H, W)  
        targets: 目标标签 (B, H, W)  
        mask: 有效区域掩码 (B, H, W)  
        ignore_index: 忽略的类别索引  
        smooth: 平滑项  
    """  
    num_classes = predictions.size(1)  
    dice_scores = []  
    
    # 将目标转换为one-hot编码  
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  
    
    # 对每个类别计算Dice系数（除了ignore_index）  
    for cls in range(num_classes):  
        if cls == ignore_index:  
            continue  
            
        pred_cls = predictions[:, cls, ...]  
        target_cls = targets_one_hot[:, cls, ...]  
        
        # 应用mask  
        pred_cls = pred_cls * mask  
        target_cls = target_cls * mask  
        
        intersection = (pred_cls * target_cls).sum()  
        union = pred_cls.sum() + target_cls.sum()  
        
        dice_score = (2. * intersection + smooth) / (union + smooth)  
        dice_scores.append(dice_score)  
    
    return torch.mean(torch.stack(dice_scores))  

def check_grad_norm(model):  
    """检查梯度范数"""  
    total_norm = 0  
    for p in model.parameters():  
        if p.grad is not None:  
            param_norm = p.grad.data.norm(2)  
            total_norm += param_norm.item() ** 2  
    total_norm = total_norm ** (1. / 2)  
    return total_norm  

def load_and_save_data(  
    image_path: str,  
    label_path: Optional[str] = None,  
    output_dir: Optional[str] = None,  
    normalize: bool = True,  
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:  
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

def calculate_iou(predictions, targets, num_classes, ignore_index=0):  
    """  
    计算IoU分数  
    
    Args:  
        predictions: 模型预测结果 (B, C, H, W)  
        targets: 目标标签 (B, H, W)  
        num_classes: 类别数量  
        ignore_index: 忽略的类别索引  
    """  
    ious = []  
    predictions = predictions.argmax(dim=1)  
    
    for cls in range(num_classes):  
        if cls == ignore_index:  
            continue  
        
        pred_mask = (predictions == cls)  
        target_mask = (targets == cls)  
        
        intersection = (pred_mask & target_mask).float().sum()  
        union = (pred_mask | target_mask).float().sum()  
        
        iou = (intersection + 1e-8) / (union + 1e-8)  
        ious.append(iou)  
    
    return torch.mean(torch.stack(ious))  