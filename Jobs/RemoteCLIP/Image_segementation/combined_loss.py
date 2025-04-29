import torch  
import torch.nn as nn  
import torch.nn.functional as F  


# 定义修改后的 FocalLoss  
class FocalLoss(nn.Module):  
    """  
    多分类的 Focal Loss 实现  
    """  
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=0):  
        super(FocalLoss, self).__init__()  
        self.gamma = gamma  
        self.alpha = alpha  
        self.ignore_index = ignore_index  

    def forward(self, inputs, targets):  
        # 将 logits 限制在 [-100, 100] 范围内，防止数值溢出  
        inputs = torch.clamp(inputs, min=-100, max=100)  

        # 计算交叉熵损失  
        ce_loss = F.cross_entropy(  
            inputs,  
            targets,  
            reduction='none',  
            ignore_index=self.ignore_index  
        )  

        # 防止 ce_loss 过大导致的数值下溢  
        ce_loss = torch.clamp(ce_loss, max=100)  

        # 计算 pt  
        pt = torch.exp(-ce_loss)  

        # 防止 pt 为 0，避免 subsequent 计算时出现 NaN  
        pt = torch.clamp(pt, min=1e-6)  

        # 计算 Focal Loss  
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  

        # 返回平均损失  
        return focal_loss.mean()  

# 定义修改后的 Dice Coefficient  
def dice_coefficient(predicts, target, epsilon=1e-5, ignore_index=0):  
    """  
    计算 Dice 系数  
    """  
    # 将 predicts 转换为概率分布  
    predicts = F.softmax(predicts, dim=1)  

    # 创建 one-hot 编码的 target  
    num_classes = predicts.shape[1]  
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes)  
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  

    # 忽略背景类别  
    valid_mask = (target != ignore_index).unsqueeze(1)  

    # 提取有效的预测和目标  
    predicts = predicts * valid_mask  
    target_one_hot = target_one_hot * valid_mask  

    # 计算交集和并集  
    intersection = torch.sum(predicts * target_one_hot, dim=(0, 2, 3))  
    cardinality = torch.sum(predicts + target_one_hot, dim=(0, 2, 3))  

    # 计算 Dice 系数，防止除零  
    dice_score = (2. * intersection + epsilon) / (cardinality + epsilon)  

    # 计算 Dice Loss  
    dice_loss = 1.0 - dice_score.mean()  

    return dice_loss  

# 定义修改后的 CombinedLoss  
class CombinedLoss(nn.Module):  
    """  
    综合 Focal Loss 和 Dice Loss 的损失函数  
    """  
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=0):  
        super(CombinedLoss, self).__init__()  
        self.focal_loss_fn = FocalLoss(gamma, alpha, ignore_index)  
        self.ignore_index = ignore_index  

    def forward(self, predicts, targets):  
        # 检查 predicts 中的数值，防止 NaN 或 Inf  
        # predicts = predicts['main']  # [B, C, H, W]  s
        
        # 在计算损失之前，处理预测值  
        predicts = torch.nan_to_num(predicts, nan=0.0, posinf=1.0, neginf=-1.0)  

        # 计算 Focal Loss  
        focal_loss = self.focal_loss_fn(predicts, targets)  

        # 计算 Dice Loss  
        dice_loss = dice_coefficient(predicts, targets, ignore_index=self.ignore_index)  

        # 在计算损失之后，检查损失值  
        if torch.isnan(focal_loss) or torch.isinf(focal_loss):  
            print("Focal Loss 出现 NaN 或 Inf，已处理。")  
            focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1.0, neginf=0.0)  

        if torch.isnan(dice_loss) or torch.isinf(dice_loss):  
            print("Dice Loss 出现 NaN 或 Inf，已处理。")  
            dice_loss = torch.nan_to_num(dice_loss, nan=0.0, posinf=1.0, neginf=0.0)  

        # 求和为总损失  
        loss = focal_loss + dice_loss  

        # 返回损失和损失信息  
        loss_info = {  
            'focal_loss': focal_loss.item(),  
            'dice_loss': dice_loss.item()  
        }  

        return loss, loss_info  