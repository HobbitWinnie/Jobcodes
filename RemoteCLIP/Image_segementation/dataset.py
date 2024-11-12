import numpy as np  
import torch  
import random  
from typing import Tuple, List, Optional, Union, Dict
from torch.utils.data import Dataset, DataLoader, random_split  
import random  
import albumentations as A  
from albumentations.pytorch import ToTensorV2  

class RemoteSensingDataset(Dataset):  
    """遥感影像数据集"""  
    def __init__(self,   
                 image: np.ndarray,  
                 labels: Optional[np.ndarray] = None,  
                 patch_size: int = 224,  
                 num_patches: int = 1000,  
                 transform = None):  
        """  
        Args:  
            image: 输入图像 [C, H, W]  
            labels: 标签图像 [H, W]  
            patch_size: 图像块大小  
            num_patches: 随机采样的图像块数量  
            transform: 数据增强转换  
            is_train: 是否为训练模式  
        """  
        self.image = image  
        self.labels = labels  
        self.patch_size = patch_size  
        self.num_patches = num_patches  

        # 确保图像格式正确  
        self.h, self.w = self.image.shape[1:]  # C, H, W  
        
        # 验证图像和标签尺寸匹配  
        if self.labels is not None:  
            assert self.h == self.labels.shape[0] and self.w == self.labels.shape[1], \
                "Image and label dimensions don't match"  
        
        # 验证图像块尺寸  
        if self.h < self.patch_size or self.w < self.patch_size:  
            raise ValueError(f"Patch size {patch_size} is too large for image size {self.h}x{self.w}")  

        # 设置数据增强  
        self.transform = transform

    def __len__(self) -> int:  
        return self.num_patches  

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  
        # 随机选择图像块位置
        max_x = self.w - self.patch_size
        max_y = self.h - self.patch_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # 提取图像块
        image_patch = self.image[:, y:y+self.patch_size, x:x+self.patch_size]
        
        # 应用数据增强
        if self.transform:
            image_patch = self.transform(image_patch)
        
        # 转换为tensor
        image_patch = torch.from_numpy(image_patch).float()
        
        # 如果有标签，同时提取标签块
        if self.labels is not None:
            label_patch = self.labels[y:y+self.patch_size, x:x+self.patch_size]
            label_patch = torch.from_numpy(label_patch).long()
            validate_labels(label_patch)
            return image_patch, label_patch
            
        return image_patch

def validate_labels(labels: torch.Tensor, num_classes: int = 9) -> None:  
    """验证标签值是否在有效范围内"""  
    unique_labels = torch.unique(labels)  
    min_label = unique_labels.min().item()  
    max_label = unique_labels.max().item()  
    if min_label < 0 or max_label >= num_classes:  
        raise ValueError(f"Labels must be in range [0, {num_classes-1}], "  
                      f"but got range [{min_label}, {max_label}]")  

def create_dataloaders(image: np.ndarray,  
                      labels: np.ndarray,  
                      patch_size: int,  
                      num_patches: int,  
                      batch_size: int,  
                      train_ratio: float = 0.8,  
                      num_workers = 4,
                      transform = None) -> Tuple[DataLoader, DataLoader]:  
    """创建训练和验证数据加载器"""  
    
    dataset = RemoteSensingDataset(  
        image=image,  
        labels=labels,  
        patch_size=patch_size,  
        num_patches=num_patches,  
        transform=transform  
    )  
    
    # 划分训练集和验证集  
    train_size = int(train_ratio * len(dataset))  
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  
    
    # 创建数据加载器  
    train_loader = DataLoader(  
        train_dataset,  
        batch_size=batch_size,  
        shuffle=True,  
        num_workers=num_workers,  
        pin_memory=True  
    )  
    
    val_loader = DataLoader(  
        val_dataset,  
        batch_size=batch_size,  
        shuffle=False,  
        num_workers=num_workers,  
        pin_memory=True  
    )  
    
    return train_loader, val_loader


def split_image_into_patches(image: np.ndarray, 
                           patch_size: int = 256, 
                           overlap: int = 128) -> List[np.ndarray]:
    """将大图像分割成重叠的小块"""
    patches = []
    c, h, w = image.shape
    stride = patch_size - overlap
    
    # 处理常规块
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    
    # 处理边缘
    if h % stride != 0:
        for x in range(0, w - patch_size + 1, stride):
            patch = image[:, h-patch_size:h, x:x+patch_size]
            patches.append(patch)
    
    if w % stride != 0:
        for y in range(0, h - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, w-patch_size:w]
            patches.append(patch)
    
    if h % stride != 0 and w % stride != 0:
        patch = image[:, h-patch_size:h, w-patch_size:w]
        patches.append(patch)
    
    return patches

def reconstruct_image_from_patches(predictions, image_size, patch_size, overlap):  
    """重建完整的预测图像"""  
    h, w = image_size  
    stride = patch_size - overlap  
    
    # 获取类别数量  
    num_classes = predictions[0].shape[0] if predictions else 1  
    
    # 初始化输出  
    reconstructed = np.zeros((h, w), dtype=np.uint8)  
    confidence = np.zeros((h, w), dtype=np.float32)  
    
    idx = 0  
    # 处理常规块  
    for y in range(0, h - patch_size + 1, stride):  
        for x in range(0, w - patch_size + 1, stride):  
            if idx >= len(predictions):  
                break  
                
            pred = predictions[idx]  
            # 确保pred是3维的 (num_classes, patch_size, patch_size)  
            if pred.ndim == 1:  
                pred = pred.reshape(num_classes, patch_size, patch_size)  
            
            # 计算每个位置的最大概率和对应的类别  
            patch_confidence = np.max(pred, axis=0)  
            patch_prediction = np.argmax(pred, axis=0)  
            
            # 更新重建图像  
            current_confidence = confidence[y:y+patch_size, x:x+patch_size]  
            update_mask = patch_confidence > current_confidence  
            
            confidence[y:y+patch_size, x:x+patch_size][update_mask] = patch_confidence[update_mask]  
            reconstructed[y:y+patch_size, x:x+patch_size][update_mask] = patch_prediction[update_mask]  
            idx += 1  
    
    # 处理边缘  
    if h % stride != 0:  
        y = h - patch_size  
        for x in range(0, w - patch_size + 1, stride):  
            if idx >= len(predictions):  
                break  
                
            pred = predictions[idx]  
            if pred.ndim == 1:  
                pred = pred.reshape(num_classes, patch_size, patch_size)  
            
            patch_confidence = np.max(pred, axis=0)  
            patch_prediction = np.argmax(pred, axis=0)  
            
            current_confidence = confidence[y:, x:x+patch_size]  
            update_mask = patch_confidence[:h-y, :] > current_confidence  
            
            confidence[y:, x:x+patch_size][update_mask] = patch_confidence[:h-y, :][update_mask]  
            reconstructed[y:, x:x+patch_size][update_mask] = patch_prediction[:h-y, :][update_mask]  
            idx += 1  
    
    if w % stride != 0:  
        x = w - patch_size  
        for y in range(0, h - patch_size + 1, stride):  
            if idx >= len(predictions):  
                break  
                
            pred = predictions[idx]  
            if pred.ndim == 1:  
                pred = pred.reshape(num_classes, patch_size, patch_size)  
            
            patch_confidence = np.max(pred, axis=0)  
            patch_prediction = np.argmax(pred, axis=0)  
            
            current_confidence = confidence[y:y+patch_size, x:]  
            update_mask = patch_confidence[:, :w-x] > current_confidence  
            
            confidence[y:y+patch_size, x:][update_mask] = patch_confidence[:, :w-x][update_mask]  
            reconstructed[y:y+patch_size, x:][update_mask] = patch_prediction[:, :w-x][update_mask]  
            idx += 1  
    
    return reconstructed  