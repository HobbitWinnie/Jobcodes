from torch.utils.data import Dataset  
from PIL import Image  
import numpy as np  
import random  

class LargeImageDataset(Dataset):  
    def __init__(self, image_path, label_path, patch_size=256, num_patches=1000, transform=None):  
        self.image = Image.open(image_path)  # 读取 TIFF 格式的影像  
        self.label = Image.open(label_path)  # 读取 TIFF 格式的标签  
        self.patch_size = patch_size  
        self.num_patches = num_patches  
        self.transform = transform  

        self.image = np.array(self.image)  
        self.label = np.array(self.label)  

        self.h, self.w, _ = self.image.shape  

    def __len__(self):  
        # 返回指定的补丁数量  
        return self.num_patches  

    def __getitem__(self, idx):  
        # 随机选择补丁的起始位置  
        max_x = self.w - self.patch_size  
        max_y = self.h - self.patch_size  
        x = random.randint(0, max_x)  
        y = random.randint(0, max_y)  

        # 提取补丁  
        image_patch = self.image[y:y+self.patch_size, x:x+self.patch_size, :]  
        label_patch = self.label[y:y+self.patch_size, x:x+self.patch_size]  

        if self.transform:  
            image_patch = Image.fromarray(image_patch)  
            label_patch = Image.fromarray(label_patch)  

            image_patch = self.transform(image_patch)  
            label_patch = self.transform(label_patch)  

        return image_patch, label_patch