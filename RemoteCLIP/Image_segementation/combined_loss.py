import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import logging  


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 0, epsilon: float = 1e-6, weight: torch.Tensor = None) -> torch.Tensor:  
    """  
    计算加权的Dice系数，忽略特定类别的像素  
    """  
    # 检查输入维度  
    assert pred.dim() == 4, f"预测tensor必须是4D (B,C,H,W)，但得到 {pred.dim()}D"  
    assert target.dim() == 3, f"目标tensor必须是3D (B,H,W)，但得到 {target.dim()}D"  
    assert pred.size(0) == target.size(0), "batch size不匹配"  

    b, c, h, w = pred.size()  

    # 忽略指定类别的像素  
    valid_mask = (target != ignore_index).float()  

    # 将目标转换为one-hot编码  
    target_one_hot = F.one_hot(  
        torch.clamp(target, 0, c - 1),  
        num_classes=c  
    ).permute(0, 3, 1, 2).float()  

    # 应用有效区域掩码  
    valid_mask = valid_mask.unsqueeze(1)  
    pred = pred * valid_mask  
    target_one_hot = target_one_hot * valid_mask  

    # 计算交集和并集  
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  
    cardinality = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  

    dice_score = (2. * intersection + epsilon) / (cardinality + epsilon)  

    # 忽略指定类别  
    if weight is not None:  
        dice_score = dice_score * weight.unsqueeze(0)  
    else:  
        class_mask = torch.ones(c, device=pred.device)  
        class_mask[ignore_index] = 0  
        dice_score = dice_score * class_mask.unsqueeze(0)  

    # 计算平均Dice系数  
    mean_dice = dice_score.sum(dim=1) / (dice_score != 0).sum(dim=1).clamp(min=1)  
    return mean_dice.mean()  

class FocalLoss(nn.Module):  
    """  
    Focal Loss，用于处理类别不平衡  
    """  
    def __init__(self, alpha=0.25, gamma=2, ignore_index=0, reduction='mean'):  
        super().__init__()  
        self.alpha = alpha  
        self.gamma = gamma  
        self.ignore_index = ignore_index  
        self.reduction = reduction  

    def forward(self, inputs, targets):  
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')  
        pt = torch.exp(-ce_loss)  # 获得预测的概率  
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  

        if self.reduction == 'mean':  
            return focal_loss.mean()  
        elif self.reduction == 'sum':  
            return focal_loss.sum()  
        else:  
            return focal_loss  

class CombinedLoss(nn.Module):  
    """  
    组合Focal Loss和Dice损失的损失函数  
    """  
    def __init__(  
        self,  
        num_classes,  
        weights = [0.4, 0.6],  
        ignore_index = 0,  
        epsilon = 1e-6,  
        reduction = 'mean',  
        alpha = 1,  
        gamma = 2,  
        class_weights = None  
    ):  
        """  
        初始化组合损失函数  
        Args:  
            num_classes: 类别数量  
            weights: 损失权重列表 [focal_weight, dice_weight]  
            ignore_index: 忽略的类别索引  
            epsilon: 防止除零的小值  
            reduction: 损失计算方式 ('mean', 'sum', 'none')  
            alpha: Focal Loss的alpha参数  
            gamma: Focal Loss的gamma参数  
            class_weights: 每个类别的权重，Tensor类型  
        """  
        super().__init__()  
        self.num_classes = num_classes  
        self.weights = weights  
        self.ignore_index = ignore_index  
        self.epsilon = epsilon  
        self.reduction = reduction  
        self.class_weights = class_weights  

        # 验证权重  
        assert len(weights) == 2, "权重必须包含两个值 [focal_weight, dice_weight]"  
        assert abs(sum(weights) - 1.0) < 1e-4, "权重之和必须为1"  

        # 初始化Focal Loss  
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index, reduction=reduction)  

        # 初始化损失历史记录  
        self.loss_history = {  
            'focal_loss': [],  
            'dice_loss': [],  
            'total_loss': []  
        }  

    def forward(self, outputs, targets):  
        device = outputs['main'].device        
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

        # 计算Focal Loss  
        focal_loss = self.focal_loss(predicts, targets)  

        # 计算Dice Loss  
        probs = F.softmax(predicts, dim=1)  
        dice_score = dice_coefficient(  
            probs,  
            targets,  
            ignore_index=self.ignore_index,  
            epsilon=self.epsilon,  
            weight=self.class_weights  
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

        # 返回结果  
        metrics = {  
            'focal_loss': focal_loss.item(),  
            'dice_loss': dice_loss.item(),  
            'total_loss': total_loss.item(),  
            'weights': self.weights,  
            'dice_score': dice_score.item()  
        }  

        return total_loss, metrics