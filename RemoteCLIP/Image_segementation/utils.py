import os  
import logging  
from typing import Tuple, Optional, Dict  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from datetime import datetime  
import rasterio  
from torchvision import transforms  

# 设置日志  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  
logger = logging.getLogger(__name__)  

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

class RomoClipLoss(nn.Module):  
    """RomoClip特定的损失函数"""  
    def __init__(self, weights=[0.5, 0.3, 0.2], ignore_index=0):  
        """  
        组合损失函数：CrossEntropy + Dice Loss + Feature loss  
        
        Args:  
            weights: 损失权重 [ce_weight, dice_weight, feature_weight]  
            ignore_index: 忽略的类别索引  
        """  
        super().__init__()  
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')  
        self.feature_loss = nn.MSELoss()  
        self.weights = weights  
        self.ignore_index = ignore_index  
        
    def forward(self, outputs, targets, clip_features=None, pred_features=None):  
        # 基础分割损失  
        mask = (targets != self.ignore_index).float()  
        
        ce_loss = self.ce(outputs['segmentation'], targets)  
        masked_ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)  
        
        dice_loss = 1 - multiclass_dice_coefficient(  
            F.softmax(outputs['segmentation'], dim=1),  
            targets,  
            mask,  
            ignore_index=self.ignore_index  
        )  
        
        # 特征匹配损失  
        feature_loss = torch.tensor(0.0).to(outputs['segmentation'].device)  
        if clip_features is not None and pred_features is not None:  
            feature_loss = self.feature_loss(pred_features, clip_features)  
        
        # 组合损失  
        total_loss = (  
            self.weights[0] * masked_ce_loss +  
            self.weights[1] * dice_loss +  
            self.weights[2] * feature_loss  
        )  
        
        return total_loss, {  
            'ce_loss': masked_ce_loss.item(),  
            'dice_loss': dice_loss.item(),  
            'feature_loss': feature_loss.item()  
        }  

def multiclass_dice_coefficient(predictions, targets, mask, ignore_index=0, smooth=1e-8):  
    """  
    计算多类别的Dice系数，排除nodata区域  
    """  
    num_classes = predictions.size(1)  
    dice_scores = []  
    
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  
    
    for cls in range(num_classes):  
        if cls == ignore_index:  
            continue  
            
        pred_cls = predictions[:, cls, ...]  
        target_cls = targets_one_hot[:, cls, ...]  
        
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

def prepare_clip_input(image: torch.Tensor) -> torch.Tensor:  
    """  
    准备RomoClip输入  
    """  
    mean = [0.48145466, 0.4578275, 0.40821073]  
    std = [0.26862954, 0.26130258, 0.27577711]  
    
    if image.shape[1] == 1:  
        image = image.repeat(1, 3, 1, 1)  
    elif image.shape[1] > 3:  
        image = image[:, :3]  
    
    if image.max() > 1:  
        image = image / 255.0  
    
    normalize = transforms.Normalize(mean=mean, std=std)  
    return normalize(image)  

def load_and_save_data(  
    image_path: str,  
    label_path: Optional[str] = None,  
    output_dir: Optional[str] = None,  
    normalize: bool = True,  
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:  
    """  
    加载并处理遥感图像和标签数据  
    """  
    if output_dir is not None:  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        output_dir = os.path.join(output_dir, timestamp)  
        os.makedirs(output_dir, exist_ok=True)  
    
    try:  
        logging.info(f"Loading image from {image_path}")  
        with rasterio.open(image_path) as src:  
            image = src.read()  
            image_nodata = int(src.nodata)  
            image_meta = src.meta  
            
        image_mask = np.ones_like(image[0], dtype=bool)  
        if image_nodata is not None:  
            image_mask = (image[0] != image_nodata)  
            
        if np.all(~image_mask):  
            raise ValueError("No valid data in image")  
            
        image = np.where(np.broadcast_to(image_mask, image.shape), image, 0)  
        
        if normalize:  
            image_normalized = np.zeros_like(image, dtype=np.float32)  
            for i in range(image.shape[0]):  
                valid_data = image[i][image_mask]  
                if len(valid_data) > 0:  
                    min_val = np.percentile(valid_data, 1)  
                    max_val = np.percentile(valid_data, 99)  
                    image_normalized[i] = np.clip((image[i] - min_val) / (max_val - min_val + 1e-8), 0, 1)  
            image = image_normalized  
            
    except Exception as e:  
        logging.error(f"Error processing image: {e}")  
        raise  
        
    labels = None  
    if label_path:  
        try:  
            logging.info(f"Loading labels from {label_path}")  
            with rasterio.open(label_path) as src:  
                labels = src.read(1)  
                labels_nodata = int(src.nodata)  
                
            label_mask = np.ones_like(labels, dtype=bool)  
            if labels_nodata is not None:  
                label_mask = (labels != labels_nodata)  
                
            labels = np.where(label_mask, labels, 0)  
                
        except Exception as e:  
            logging.error(f"Error processing labels: {e}")  
            raise  
    
    logging.info(f"Image shape: {image.shape}, dtype: {image.dtype}")  
    if labels is not None:  
        logging.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")  
        
    return image, labels, image_meta  

def calculate_iou(predictions, targets, num_classes, ignore_index=0):  
    """计算IoU分数"""  
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

def evaluate_model(model, dataloader, device, num_classes):  
    """评估模型性能"""  
    model.eval()  
    metrics = {  
        'iou': [],  
        'dice': [],  
        'accuracy': [],  
        'feature_similarity': []  
    }  
    
    with torch.no_grad():  
        for batch in dataloader:  
            images, targets = batch  
            images, targets = images.to(device), targets.to(device)  
            
            outputs = model(images)  
            
            iou = calculate_iou(outputs['segmentation'], targets, num_classes)  
            metrics['iou'].append(iou.item())  
            
            dice = multiclass_dice_coefficient(  
                F.softmax(outputs['segmentation'], dim=1),  
                targets,  
                torch.ones_like(targets).float(),  
                ignore_index=0  
            )  
            metrics['dice'].append(dice.item())  
            
            pred = outputs['segmentation'].argmax(dim=1)  
            accuracy = (pred == targets).float().mean()  
            metrics['accuracy'].append(accuracy.item())  
            
            if 'features' in outputs:  
                similarity = F.cosine_similarity(  
                    outputs['features'],  
                    outputs['clip_features']  
                ).mean()  
                metrics['feature_similarity'].append(similarity.item())  
    
    results = {  
        k: np.mean(v) for k, v in metrics.items() if v  
    }  
    
    return results
