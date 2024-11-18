import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import logging  

def multiclass_dice_coefficient(predictions, targets, mask, ignore_index=0, smooth=1e-8):  
    """  
    计算多类别的Dice系数，排除nodata区域  
    
    Args:  
        predictions: 模型预测结果 (B, C, H, W)  
        targets: 目标标签 (B, H, W)  
        mask: 有效区域掩码 (B, H, W)  
        ignore_index: 忽略的类别索引  
        smooth: 平滑项  
    
    Returns:  
        torch.Tensor: 平均Dice系数  
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

class CombinedLoss(nn.Module):  
    """  
    组合损失函数：CrossEntropy + Dice Loss  
    
    Args:  
        weights: 两种损失的权重 [ce_weight, dice_weight]，默认[0.5, 0.5]  
        ignore_index: 忽略的类别索引，默认为0（nodata）  
        smooth: 平滑项，防止除零，默认1e-8  
        reduction: 损失计算方式，默认'mean'  
    """  
    def __init__(  
        self,   
        weights=[0.5, 0.5],   
        ignore_index=0,   
        smooth=1e-8,  
        reduction='mean'  
    ):  
        super().__init__()  
        self.ce = nn.CrossEntropyLoss(  
            ignore_index=ignore_index,   
            reduction='none'  
        )  
        self.weights = weights  
        self.ignore_index = ignore_index  
        self.smooth = smooth  
        self.reduction = reduction  
        
        # 验证权重  
        assert len(weights) == 2, "权重必须包含两个值 [ce_weight, dice_weight]"  
        assert abs(sum(weights) - 1.0) < 1e-4, "权重之和必须为1"  
        
    def forward(self, outputs, targets, progress):  
        """  
        计算组合损失  
        
        Args:  
            outputs: 模型输出 (B, C, H, W)  
            targets: 目标标签 (B, H, W)  
            
        Returns:  
            tuple: (total_loss, loss_dict)  
                - total_loss: 总损失值  
                - loss_dict: 包含各个损失组件的字典  
        """  

        if progress < 0.3:  
            self.weights = [0.7, 0.3]  
            
        # 中期：平衡两种损失  
        elif progress < 0.6:  
            self.weights = [0.5, 0.5]  
            
        # 后期：更注重整体形状  
        else:  
            self.weights = [0.3, 0.7]  
                
        # if isinstance(outputs, dict):  
        #     outputs = outputs['main'] if 'main' in outputs else outputs['out']  
        # else:  
        #     outputs = outputs  
        
        # 基本的数值检查和处理  
        if torch.isnan(outputs['main']).any() or torch.isinf(outputs['main']).any():  
            outputs = torch.nan_to_num(  
                outputs,   
                nan=0.0,   
                posinf=1e7,   
                neginf=-1e7  
            )  
            logging.warning("检测到NaN或Inf值，已进行处理")  
        
        # 创建有效区域掩码  
        mask = (targets != self.ignore_index).float()  
        predicts = outputs['main']
        # 计算CE损失  

        ce_loss = self.ce(predicts, targets)  
        if self.reduction == 'mean':  
            ce_loss = (ce_loss * mask).sum() / (mask.sum() + self.smooth)  
        else:  # sum  
            ce_loss = (ce_loss * mask).sum()  
        
        # 计算Dice损失  
        dice_loss = 1 - multiclass_dice_coefficient(  
            F.softmax(predicts, dim=1),  
            targets,  
            mask,  
            ignore_index=self.ignore_index,  
            smooth=self.smooth  
        )  
        
        # 组合损失  
        total_loss = self.weights[0] * ce_loss + self.weights[1] * dice_loss  
        
        # 检查损失值的有效性  
        if torch.isnan(total_loss) or torch.isinf(total_loss):  
            logging.error(f"损失计算异常! CE Loss: {ce_loss.item()}, Dice Loss: {dice_loss.item()}")  
            raise ValueError("损失计算结果无效")  
        
        # 返回总损失和详细信息  
        return total_loss, {  
            'ce_loss': ce_loss.item(),  
            'dice_loss': dice_loss.item(),  
            'total_loss': total_loss.item(),  
            'weights': self.weights  
        }  
        
    def update_weights(self, new_weights):  
        """  
        更新损失权重  
        
        Args:  
            new_weights: 新的权重 [ce_weight, dice_weight]  
        """  
        assert len(new_weights) == 2, "权重必须包含两个值"  
        assert abs(sum(new_weights) - 1.0) < 1e-4, "权重之和必须为1"  
        self.weights = new_weights  
        logging.info(f"损失权重已更新为: {new_weights}")