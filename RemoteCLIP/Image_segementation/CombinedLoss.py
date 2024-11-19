import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import logging  
import numpy as np  

def dice_coefficient(  
    pred: torch.Tensor,  
    target: torch.Tensor,  
    ignore_index: int = 0,  
    epsilon: float = 1e-6  
) -> torch.Tensor:  
    """  
    计算Dice系数，忽略特定类别的像素  
    
    Args:  
        pred: (B, C, H, W) softmax后的预测概率  
        target: (B, H, W) 目标标签  
        ignore_index: 忽略的类别索引  
        epsilon: 防止除零的小值  
    
    Returns:  
        torch.Tensor: 平均Dice系数（忽略ignore_index类别）  
    """  
    assert pred.dim() == 4, f"预测tensor必须是4D (B,C,H,W), 得到 {pred.dim()}D"  
    assert target.dim() == 3, f"目标tensor必须是3D (B,H,W), 得到 {target.dim()}D"  
    assert pred.size(0) == target.size(0), "batch size不匹配"  
    
    b, c, h, w = pred.size()  
    
    # 创建有效区域掩码  
    valid_mask = (target != ignore_index).float()  
    
    # 转换target为one-hot编码，只处理有效区域  
    target_one_hot = F.one_hot(  
        torch.where(valid_mask == 1, target, 0),  
        num_classes=c  
    ).permute(0, 3, 1, 2).float()  
    
    # 扩展mask到所有通道  
    valid_mask = valid_mask.unsqueeze(1).expand_as(pred)  
    
    # 应用mask到预测和目标  
    pred = pred * valid_mask  
    target_one_hot = target_one_hot * valid_mask  
    
    # 计算每个类别的dice系数  
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  
    cardinality = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  
    
    # 计算每个类别的dice系数  
    dice = (2. * intersection + epsilon) / (cardinality + epsilon)  
    
    # 创建类别掩码，排除ignore_index类别  
    class_mask = torch.ones((b, c), device=pred.device)  
    class_mask[:, ignore_index] = 0  
    
    # 计算有效类别的平均dice系数  
    valid_dice = dice * class_mask  
    num_valid_classes = class_mask.sum(dim=1, keepdim=True).clamp(min=1)  
    mean_dice = (valid_dice.sum(dim=1) / num_valid_classes.squeeze()).mean()  
    
    return mean_dice  

class CombinedLoss(nn.Module):  
    """  
    组合BCE和Dice损失的损失函数  
    """  
    def __init__(  
        self,  
        num_classes: int,  
        class_weights: torch.Tensor = None,  
        weights: list = [0.5, 0.5],  
        ignore_index: int = 0,  
        epsilon: float = 1e-6,  
        reduction: str = 'mean'  
    ):  
        """  
        初始化组合损失函数  
        
        Args:  
            num_classes: 类别数量  
            class_weights: 每个类别的权重  
            weights: BCE和Dice损失的权重 [bce_weight, dice_weight]  
            ignore_index: 忽略的类别索引  
            epsilon: 防止除零的小值  
            reduction: 损失计算方式 ('mean', 'sum', 'none')  
        """  
        super().__init__()  
        self.num_classes = num_classes  
        
        # 初始化类别权重  
        if class_weights is None:  
            class_weights = torch.ones(num_classes)  
        elif not isinstance(class_weights, torch.Tensor):  
            class_weights = torch.FloatTensor(class_weights)  
        
        self.register_buffer('class_weights', class_weights)  
        
        # 初始化BCE损失  
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)  
        
        self.weights = weights  
        self.ignore_index = ignore_index  
        self.epsilon = epsilon  
        self.reduction = reduction  
        
        # 验证权重  
        assert len(weights) == 2, "权重必须包含两个值 [bce_weight, dice_weight]"  
        assert abs(sum(weights) - 1.0) < 1e-4, "权重之和必须为1"  
        
        # 初始化损失历史记录  
        self.loss_history = {  
            'bce_loss': [],  
            'dice_loss': [],  
            'total_loss': []  
        }  

    def update_class_weights(self, new_weights):  
        """  
        更新类别权重  
        
        Args:  
            new_weights: 新的类别权重  
        """  
        device = self.class_weights.device  
        if not isinstance(new_weights, torch.Tensor):  
            new_weights = torch.FloatTensor(new_weights)  
        new_weights = new_weights.to(device)  
        self.class_weights.copy_(new_weights)  
        logging.info(f"类别权重已更新: {new_weights}")  

    def get_dynamic_weights(self, progress):  
        """  
        根据训练进度动态调整损失权重  
        
        Args:  
            progress: 训练进度 (0~1)  
        
        Returns:  
            list: [bce_weight, dice_weight]  
        """  
        if progress < 0.2:  
            return [0.7, 0.3]  
        elif progress < 0.5:  
            return [0.5, 0.5]  
        elif progress < 0.8:  
            return [0.3, 0.7]  
        else:  
            return [0.2, 0.8]  

    def forward(self, outputs, targets, progress):  
        """  
        计算组合损失  
        
        Args:  
            outputs: 模型输出的字典，包含 'main' 键  
            targets: 目标标签  
            progress: 训练进度 (0~1)  
            
        Returns:  
            tuple: (total_loss, metrics_dict)  
        """  
        device = outputs['main'].device  
        self.weights = self.get_dynamic_weights(progress)  
        
        predicts = outputs['main']  # [B, C, H, W]  
        
        # 检查设备一致性  
        if predicts.device != targets.device:  
            logging.warning(f"设备不匹配! predicts: {predicts.device}, targets: {targets.device}")  
            targets = targets.to(device)  
        
        # 处理数值问题  
        if torch.isnan(predicts).any() or torch.isinf(predicts).any():  
            predicts = torch.nan_to_num(  
                predicts,  
                nan=0.0,  
                posinf=1e7,  
                neginf=-1e7  
            )  
            logging.warning("检测到NaN或Inf值，已处理")  
        
        # 获取形状信息  
        b, c, h, w = predicts.size()  
        
        # 创建mask用于忽略特定类别  
        valid_mask = (targets != self.ignore_index).float()  
        
        # 准备BCE的目标tensor - 改用展平的方式处理  
        targets_flat = targets.view(-1)  # [B*H*W]  
        valid_mask_flat = valid_mask.view(-1)  # [B*H*W]  
        
        # 创建one-hot编码  
        target_one_hot = F.one_hot(  
            torch.where(valid_mask_flat == 1, targets_flat, 0),  
            num_classes=self.num_classes  
        ).float()  # [B*H*W, C]  
        
        # 重塑预测tensor  
        predicts_flat = predicts.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]  
        predicts_flat = predicts_flat.view(-1, self.num_classes)  # [B*H*W, C]  
        
        # 应用mask  
        valid_samples = valid_mask_flat == 1  
        predicts_valid = predicts_flat[valid_samples]  
        targets_valid = target_one_hot[valid_samples]  
        
        # 计算BCE loss  
        bce_loss = self.bce(predicts_valid, targets_valid)  
        
        # 计算dice loss  
        dice_score = dice_coefficient(  
            F.softmax(predicts, dim=1),  
            targets,  
            ignore_index=self.ignore_index,  
            epsilon=self.epsilon  
        )  
        dice_loss = 1 - dice_score  
        
        # 组合损失  
        total_loss = self.weights[0] * bce_loss + self.weights[1] * dice_loss  
        
        # 记录历史  
        self.loss_history['bce_loss'].append(bce_loss.item())  
        self.loss_history['dice_loss'].append(dice_loss.item())  
        self.loss_history['total_loss'].append(total_loss.item())  
        
        # 检查异常值  
        if torch.isnan(total_loss) or torch.isinf(total_loss):  
            logging.error(  
                f"损失计算异常! BCE Loss: {bce_loss.item()}, "  
                f"Dice Loss: {dice_loss.item()}"  
            )  
            raise ValueError("损失计算结果无效")  
        
        # 计算类别准确率  
        pred_classes = predicts.argmax(1)  
        class_accuracies = []  
        for i in range(self.num_classes):  
            if i == self.ignore_index:  
                continue  
            mask = targets == i  
            if mask.sum() > 0:  
                acc = (pred_classes[mask] == targets[mask]).float().mean()  
                class_accuracies.append((i, acc.item()))  
        
        return total_loss, {  
            'bce_loss': bce_loss.item(),  
            'dice_loss': dice_loss.item(),  
            'total_loss': total_loss.item(),  
            'weights': self.weights,  
            'dice_score': dice_score.item(),  
            'class_accuracies': class_accuracies  
        }  

    def get_loss_statistics(self):  
        """  
        获取损失统计信息  
        
        Returns:  
            dict: 包含各类损失的统计信息  
        """  
        stats = {}  
        for loss_type, values in self.loss_history.items():  
            if values:  
                stats[loss_type] = {  
                    'mean': np.mean(values),  
                    'std': np.std(values),  
                    'min': np.min(values),  
                    'max': np.max(values)  
                }  
        return stats