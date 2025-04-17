import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np  

# 数据集类
class WHURS19Dataset(Dataset):
    def __init__(self, data_path, preprocess_func):
        self.preprocess_func = preprocess_func
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_path))

        for cls in self.classes:
            class_dir = os.path.join(data_path, cls)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.image_paths.append(os.path.join(class_dir, fname))
                        self.labels.append(cls)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.preprocess_func is not None:
            image = self.preprocess_func(image)  
        
        else:
            image = image.resize((224, 224))
            
            # 转换为numpy数组并调整通道顺序 (H,W,C) -> (C,H,W)  
            image = np.array(image).transpose(2, 0, 1)  
            image = torch.from_numpy(image).float() 
        
        return image, label, image_path
