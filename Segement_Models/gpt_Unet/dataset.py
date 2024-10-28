import numpy as np  
import torch  
from torch.utils.data import Dataset  
import random  
import logging


def split_image_into_patches(image, patch_size=256, overlap=128):  
    """  
    将图像分割为patches  
    
    Args:  
        image (np.ndarray): 输入图像，形状为 (C, H, W)  
        patch_size (int): patch的大小  
        overlap (int): 重叠区域的大小  
    
    Returns:  
        patches (list): patch列表  
        indices (list): 每个patch的位置信息，格式为 [(y, x), ...]  
    """  
    patches = []  
    indices = []  
    C, H, W = image.shape  
    stride = patch_size - overlap  
    
    for y in range(0, H - patch_size + 1, stride):  
        for x in range(0, W - patch_size + 1, stride):  
            patch = image[:, y:y + patch_size, x:x + patch_size]  
            patches.append(patch)  
            indices.append((y, x))  
            
    logging.info(f"Split image into {len(patches)} patches with size {patch_size} and overlap {overlap}")  
    return patches, indices  

def reconstruct_image_from_patches(predictions, indices, original_shape, patch_size, overlap):  
    """  
    重建完整的预测图像，使用最高置信度策略处理重叠区域  
    
    Args:  
        predictions (list): patch预测结果列表  
        indices (list): 每个patch的位置信息  
        original_shape (tuple): 原始图像形状 (C, H, W)  
        patch_size (int): patch的大小  
        overlap (int): 重叠区域的大小  
    
    Returns:  
        np.ndarray: 重建后的完整图像  
    """  
    _, h, w = original_shape  
    reconstructed = np.zeros((h, w), dtype=np.uint8)  
    confidence = np.zeros((h, w), dtype=np.float32)  
    
    for pred, (y, x) in zip(predictions, indices):  
        # 计算实际的patch区域大小（处理边界情况）  
        y_end = min(y + patch_size, h)  
        x_end = min(x + patch_size, w)  
        patch_height = y_end - y  
        patch_width = x_end - x  
        
        # 确保预测结果的形状正确  
        if isinstance(pred, np.ndarray):  
            if len(pred.shape) == 2:  
                # 如果预测结果已经是类别标签，直接使用  
                patch_prediction = pred[:patch_height, :patch_width]  
                patch_confidence = np.ones((patch_height, patch_width), dtype=np.float32)  
            else:  
                # 如果预测结果是概率分布，计算最大概率和对应的类别  
                patch_confidence = np.max(pred[:patch_height, :patch_width], axis=0)  
                patch_prediction = np.argmax(pred[:patch_height, :patch_width], axis=0)  
        else:  
            logging.warning(f"Unexpected prediction type: {type(pred)}")  
            continue  
        
        # 使用最高置信度策略更新预测  
        update_mask = patch_confidence > confidence[y:y_end, x:x_end]  
        confidence[y:y_end, x:x_end][update_mask] = patch_confidence[update_mask]  
        reconstructed[y:y_end, x:x_end][update_mask] = patch_prediction[update_mask]  
    
    logging.info(f"Reconstructed image shape: {reconstructed.shape}")  
    return reconstructed


def validate_labels(labels, num_classes=9):  
    """验证标签值是否在有效范围内"""  
    unique_labels = torch.unique(labels)  
    min_label = unique_labels.min().item()  
    max_label = unique_labels.max().item()  
    if min_label < 0 or max_label >= num_classes:  
        raise ValueError(f"Labels must be in range [0, {num_classes-1}], "  
                        f"but got range [{min_label}, {max_label}]")


class RemoteSensingDataset(Dataset):  
    def __init__(self, image, labels, patch_size=256, num_patches=1000):  
        self.image = image  
        self.labels = labels  
        self.patch_size = patch_size  
        self.num_patches = num_patches  

        # Ensure the image and labels have the correct dimensions  
        self.image = np.transpose(self.image, (1, 2, 0))  # [H, W, C]  
        self.labels = np.expand_dims(self.labels, axis=0)  # [1, H, W]  

        self.h, self.w, _ = self.image.shape  

    def __len__(self):  
        return self.num_patches  

    def __getitem__(self, idx):  
        # Ensure patch size is not larger than the image dimensions  
        if self.h < self.patch_size or self.w < self.patch_size:  
            raise ValueError(f"Patch size {self.patch_size} is too large for image size {self.h}x{self.w}")  

        # Randomly select the starting position of the patch  
        max_x = self.w - self.patch_size  
        max_y = self.h - self.patch_size  
        x = random.randint(0, max_x)  
        y = random.randint(0, max_y)  

        # Extract the patch  
        image_patch = self.image[y:y+self.patch_size, x:x+self.patch_size, :]  
        label_patch = self.labels[:, y:y+self.patch_size, x:x+self.patch_size]  

        # Convert to tensor  
        image_patch = torch.from_numpy(image_patch.transpose((2, 0, 1))).float()  
        label_patch = torch.from_numpy(label_patch.squeeze()).long()  

        validate_labels(label_patch)  

        return image_patch, label_patch