import numpy as np  
import torch  
from torch.utils.data import Dataset  
import random  
import torchvision.transforms.functional as TF  
import cv2  # 用于图像处理的OpenCV  


def split_image_into_patches(image, patch_size=256, overlap=128):  
    patches = []  
    C, H, W = image.shape  
    step = patch_size - overlap  
    for i in range(0, H - patch_size + 1, step):  
        for j in range(0, W - patch_size + 1, step):  
            patch = image[:, i:i + patch_size, j:j + patch_size]  
            patches.append(patch)  
    return patches  

def reconstruct_image_from_patches(patches, image_shape, patch_size=256, overlap=128):  
    C, H, W = image_shape  
    reconstructed_image = np.zeros((H, W), dtype=np.int32)  
    patch_count = np.zeros((H, W), dtype=np.int32)  

    step = patch_size - overlap  
    patch_idx = 0  
    for i in range(0, H - patch_size + 1, step):  
        for j in range(0, W - patch_size + 1, step):  
            patch = patches[patch_idx]  
            for x in range(patch_size):  
                for y in range(patch_size):  
                    pixel_value = patch[x, y]  
                    # Update the reconstructed image with majority voting  
                    if patch_count[i + x, j + y] == 0:  
                        reconstructed_image[i + x, j + y] = pixel_value  
                    else:  
                        # If there's a tie, keep the existing value  
                        if np.count_nonzero(reconstructed_image[i + x, j + y] == pixel_value) > patch_count[i + x, j + y] / 2:  
                            reconstructed_image[i + x, j + y] = pixel_value  
                    patch_count[i + x, j + y] += 1  
            patch_idx += 1  

    return reconstructed_image  

def apply_augmentations(image_patch, label_patch, mask_patch):  
    """对图像、标签和掩码块应用随机增强。"""  
    # 水平翻转  
    if random.random() > 0.5:  
        image_patch = np.flip(image_patch, axis=1).copy()  
        label_patch = np.flip(label_patch, axis=2).copy()  
        mask_patch = np.flip(mask_patch, axis=2).copy()  

    # 垂直翻转  
    if random.random() > 0.5:  
        image_patch = np.flip(image_patch, axis=0).copy()  
        label_patch = np.flip(label_patch, axis=1).copy()  
        mask_patch = np.flip(mask_patch, axis=1).copy()  

    # 随机旋转  
    angle = random.choice([0, 90, 180, 270])  
    if angle != 0:  
        image_patch = np.rot90(image_patch, k=angle // 90, axes=(0, 1)).copy()  
        label_patch = np.rot90(label_patch, k=angle // 90, axes=(1, 2)).copy()  
        mask_patch = np.rot90(mask_patch, k=angle // 90, axes=(1, 2)).copy()  

    # 随机亮度调整  
    if random.random() > 0.5:  
        brightness_factor = random.uniform(0.8, 1.2)  
        image_patch = np.clip(image_patch * brightness_factor, 0, 255)  

    # 随机对比度调整  
    if random.random() > 0.5:  
        contrast_factor = random.uniform(0.8, 1.2)  
        mean = np.mean(image_patch, axis=(0, 1), keepdims=True)  
        image_patch = np.clip((image_patch - mean) * contrast_factor + mean, 0, 255)  

    return image_patch, label_patch, mask_patch  

class RemoteSensingDataset(Dataset):  
    """用于遥感数据的数据集类，处理图像、标签和数据增强。"""  

    def __init__(self, image, labels, labels_nodata=0, patch_size=256, num_patches=1000):  
        self.image = image  
        self.labels = labels  
        self.labels_nodata = labels_nodata  
        self.patch_size = patch_size  
        self.num_patches = num_patches  

        # 确保图像和标签具有正确的维度  
        self.image = np.transpose(self.image, (1, 2, 0))  # [H, W, C]  
        self.labels = np.expand_dims(self.labels, axis=0)  # [1, H, W]  

        self.h, self.w, _ = self.image.shape  

    def __len__(self):  
        return self.num_patches  

    def __getitem__(self, idx):  
        if self.h < self.patch_size or self.w < self.patch_size:  
            raise ValueError(f"Patch size {self.patch_size} is too large for image size {self.h}x{self.w}")  

        # 随机选择一个块的位置  
        max_x = self.w - self.patch_size  
        max_y = self.h - self.patch_size  
        x = random.randint(0, max_x)  
        y = random.randint(0, max_y)  

        # 提取块  
        image_patch = self.image[y:y + self.patch_size, x:x + self.patch_size, :]  
        label_patch = self.labels[:, y:y + self.patch_size, x:x + self.patch_size]  
        mask_patch = (label_patch != self.labels_nodata).astype(np.float32)  

        # 应用增强  
        image_patch, label_patch, mask_patch = apply_augmentations(image_patch, label_patch, mask_patch)  

        # 转换为张量  
        image_patch = torch.tensor(image_patch.transpose(2, 0, 1), dtype=torch.float32)  # [C, H, W]  
        label_patch = torch.tensor(label_patch.squeeze(), dtype=torch.long)  
        mask_patch = torch.tensor(mask_patch.squeeze(), dtype=torch.float32)  

        return image_patch, label_patch, mask_patch 