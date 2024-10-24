import numpy as np  
import torch  
from torch.utils.data import Dataset  
import random  


def split_image_into_patches(image, patch_size=256, overlap=128):  
    patches = []  
    C, H, W = image.shape  
    step = patch_size - overlap  
    for i in range(0, H - patch_size + 1, step):  
        for j in range(0, W - patch_size + 1, step):  
            patch = image[:, i:i + patch_size, j:j + patch_size]  
            patches.append(patch)  
    return patches  


def reconstruct_image_from_patches(predictions, original_shape, patch_size, overlap):  
    """重建完整的预测图像，使用最高置信度策略处理重叠区域"""  
    _, h, w = original_shape  # 假设 original_shape 是(C, H, W)  
    reconstructed = np.zeros((h, w), dtype=np.uint8)  
    confidence = np.zeros((h, w), dtype=np.float32)  

    stride = patch_size - overlap  
    for i, patch in enumerate(predictions):  
        y = (i * stride) // w * stride  
        x = (i * stride) % w  

        y_end = min(y + patch_size, h)  
        x_end = min(x + patch_size, w)  
        patch_height, patch_width = y_end - y, x_end - x  
        patch_confidence = np.max(patch[:, :patch_height, :patch_width], axis=0)  
        patch_prediction = np.argmax(patch[:, :patch_height, :patch_width], axis=0)  

        # 使用最高置信度策略更新预测  
        update_mask = patch_confidence > confidence[y:y_end, x:x_end]  
        confidence[y:y_end, x:x_end][update_mask] = patch_confidence[update_mask]  
        reconstructed[y:y_end, x:x_end][update_mask] = patch_prediction[update_mask]  

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