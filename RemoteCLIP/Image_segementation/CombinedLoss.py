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
    
    Notes:  
        - 输入pred应该已经经过softmax处理  
        - 返回值是所有非ignore类别的平均Dice系数  
    """  
    # 输入验证  
    assert pred.dim() == 4, f"预测tensor必须是4D (B,C,H,W), 得到 {pred.dim()}D"  
    assert target.dim() == 3, f"目标tensor必须是3D (B,H,W), 得到 {target.dim()}D"  
    assert pred.size(0) == target.size(0), "batch size不匹配"  
    
    # 获取维度信息  
    b, c, h, w = pred.size()  
    
    # 创建有效区域掩码  
    valid_mask = (target != ignore_index).float()  
    
    # 转换target为one-hot编码，只处理有效区域  
    target_one_hot = F.one_hot(  
        torch.where(valid_mask == 1, target, 0),  # 将ignore_index区域设为0  
        num_classes=c  
    ).permute(0, 3, 1, 2).float()  
    
    # 扩展mask到所有通道  
    valid_mask = valid_mask.unsqueeze(1).expand_as(pred)  
    
    # 应用mask到预测和目标  
    pred = pred * valid_mask  
    target_one_hot = target_one_hot * valid_mask  
    
    # 计算每个类别的dice系数  
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  # [B, C]  
    cardinality = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # [B, C]  
    
    # 计算每个类别的dice系数  
    dice = (2. * intersection + epsilon) / (cardinality + epsilon)  # [B, C]  
    
    # 创建类别掩码，排除ignore_index类别  
    class_mask = torch.ones((b, c), device=pred.device)  
    class_mask[:, ignore_index] = 0  
    
    # 计算有效类别的平均dice系数  
    valid_dice = dice * class_mask  
    num_valid_classes = class_mask.sum(dim=1, keepdim=True).clamp(min=1)  # 防止除零  
    mean_dice = (valid_dice.sum(dim=1) / num_valid_classes.squeeze()).mean()  
    
    return mean_dice  

class FocalLoss(nn.Module):  
    def __init__(self, alpha=None, gamma=2, ignore_index=0, reduction='mean'):  
        super().__init__()  
        self.gamma = gamma  
        self.ignore_index = ignore_index  
        self.reduction = reduction  
        
        # 初始化时将alpha转换为tensor但不指定设备  
        if alpha is not None:  
            if not isinstance(alpha, torch.Tensor):  
                alpha = torch.FloatTensor(alpha)  
            # 注册为buffer但不立即指定设备  
            self.register_buffer('alpha', alpha.clone())  
        else:  
            self.register_buffer('alpha', None)  

    def forward(self, inputs, targets):  
        # 确保在正确的设备上  
        device = inputs.device  
        
        # 计算CE损失  
        ce_loss = F.cross_entropy(  
            inputs, targets,   
            reduction='none',   
            ignore_index=self.ignore_index  
        )  
        
        # 计算focal loss的pt  
        pt = torch.exp(-ce_loss)  
        focal_loss = (1 - pt) ** self.gamma * ce_loss  

        # 处理alpha权重  
        if self.alpha is not None:  
            # 确保alpha在正确的设备上  
            alpha = self.alpha.to(device)  
            # 使用索引张量必须为long类型  
            alpha_weight = alpha[targets.long()]  
            focal_loss = alpha_weight * focal_loss  

        # 根据reduction方式返回  
        if self.reduction == 'mean':  
            return focal_loss.mean()  
        return focal_loss.sum()  

class CombinedLoss(nn.Module):  
    def __init__(  
        self,  
        num_classes,  
        class_weights=None,  
        weights=[0.5, 0.5],  
        ignore_index=0,  
        focal_gamma=2,  
        epsilon=1e-6,  
        reduction='mean'  
    ):  
        super().__init__()  
        self.num_classes = num_classes  
        
        # 初始化类别权重  
        if class_weights is None:  
            class_weights = torch.ones(num_classes)  
        elif not isinstance(class_weights, torch.Tensor):  
            class_weights = torch.FloatTensor(class_weights)  
            
        self.register_buffer('class_weights', class_weights)  
        
        # 初始化FocalLoss  
        self.focal = FocalLoss(  
            alpha=class_weights,  # 直接传入tensor  
            gamma=focal_gamma,  
            ignore_index=ignore_index,  
            reduction=reduction  
        )  
        
        self.weights = weights  
        self.ignore_index = ignore_index  
        self.epsilon = epsilon  
        self.reduction = reduction  
        
        # 验证权重  
        assert len(weights) == 2, "权重必须包含两个值 [focal_weight, dice_weight]"  
        assert abs(sum(weights) - 1.0) < 1e-4, "权重之和必须为1"  
        
        self.loss_history = {  
            'focal_loss': [],  
            'dice_loss': [],  
            'total_loss': []  
        }  

    def update_class_weights(self, new_weights):  
        """更新类别权重"""  
        device = self.class_weights.device  
        if not isinstance(new_weights, torch.Tensor):  
            new_weights = torch.FloatTensor(new_weights)  
        new_weights = new_weights.to(device)  
        self.class_weights.copy_(new_weights)  
        # 更新focal loss中的alpha  
        self.focal.alpha = new_weights  
        logging.info(f"类别权重已更新: {new_weights}")  

    def get_dynamic_weights(self, progress):  
        """根据训练进度动态调整损失权重"""  
        if progress < 0.2:  
            return [0.8, 0.2]  
        elif progress < 0.5:  
            return [0.6, 0.4]  
        elif progress < 0.8:  
            return [0.4, 0.6]  
        else:  
            return [0.3, 0.7]  

    def forward(self, outputs, targets, progress):  
        """计算组合损失"""  
        device = outputs['main'].device  
        self.weights = self.get_dynamic_weights(progress)  
        
        predicts = outputs['main']  
        
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
        
        # 计算focal loss  
        focal_loss = self.focal(predicts, targets)  
        
        # 计算dice loss  
        dice_score = dice_coefficient(  
            F.softmax(predicts, dim=1),  
            targets,  
            ignore_index=self.ignore_index,  
            epsilon=self.epsilon  
        )  
        dice_loss = 1 - dice_score  
        
        # 组合损失  
        total_loss = self.weights[0] * focal_loss + self.weights[1] * dice_loss  
        
        # 记录历史  
        self.loss_history['focal_loss'].append(focal_loss.item())  
        self.loss_history['dice_loss'].append(dice_loss.item())  
        self.loss_history['total_loss'].append(total_loss.item())  
        
        # 检查异常值  
        if torch.isnan(total_loss) or torch.isinf(total_loss):  
            logging.error(  
                f"损失计算异常! Focal Loss: {focal_loss.item()}, "  
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
            'focal_loss': focal_loss.item(),  
            'dice_loss': dice_loss.item(),  
            'total_loss': total_loss.item(),  
            'weights': self.weights,  
            'dice_score': dice_score.item(),  
            'class_accuracies': class_accuracies  
        }  

    def get_loss_statistics(self):  
        """获取损失统计信息"""  
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