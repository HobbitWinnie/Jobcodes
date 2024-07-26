import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class FocalLoss(nn.Module):  
    """  
    Focal Loss是一种改进的交叉熵损失 用于处理不平衡类别问题, 
    它通过向难分类的样本赋予更高的权重来减少易分类样本对总损失的贡献
    
    参数:  
    - alpha: 平衡因子，为每个样本加权。  
    - gamma: 调整易分类样本的损失权重。  
    - logits: 指示输入是未激活的logits还是已激活的概率。  
    - reduce: 是否对结果进行均值化。  
    """  
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):  
        super(FocalLoss, self).__init__()  
        self.alpha = alpha  
        self.gamma = gamma  
        self.logits = logits  
        self.reduce = reduce  
        
    def forward(self, inputs, targets):  
        if self.logits:  
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  
        else:  
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')  
        
        pt = torch.exp(-BCE_loss)  
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  
        
        if self.reduce:  
            return torch.mean(F_loss)  
        else:  
            return F_loss  

class DiceLoss(nn.Module):  
    """  
    Dice Loss 主要用于语义分割，但对于多标签分类任务同样有效, 通过测量预测结果和真实标签之间的重叠面积来优化模型。
    
    参数:  
    - smooth: 用于平滑的常数，避免分母为0。  
    """  
    def __init__(self):  
        super(DiceLoss, self).__init__()  

    def forward(self, inputs, targets, smooth=1):  
        # Sigmoid 激活输入，使之处于 0-1 之间  
        inputs = torch.sigmoid(inputs)  
        # 将张量展平  
        inputs = inputs.view(-1)  
        targets = targets.view(-1)  
        
        intersection = (inputs * targets).sum()  
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice  

class LabelSmoothingLoss(nn.Module):  
    """  
    标签平滑损失用于减轻模型对训练数据的过拟合。  
    
    参数:  
    - classes: 类别数目。  
    - smoothing: 平滑因子，减少置信度过高的预测。  
    """  
    def __init__(self, classes, smoothing=0.0):  
        super(LabelSmoothingLoss, self).__init__()  
        self.confidence = 1.0 - smoothing  
        self.smoothing = smoothing  
        self.cls = classes  
        self.criterion = nn.KLDivLoss(reduction='batchmean')  

    def forward(self, output, target):  
        batch_size, num_classes = output.size()  
        # 创建一个平滑标签张量  
        smooth_label = torch.full((batch_size, num_classes), self.smoothing / (num_classes-1)).to(output.device)  
        # 将目标标记的索引位置设为置信度  
        target_one_hot = smooth_label.scatter(1, target.unsqueeze(1), self.confidence)  
        output = F.log_softmax(output, dim=1)  
        return self.criterion(output, target_one_hot)