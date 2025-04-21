import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split


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
    

def get_loader(
        data_path, 
        preprocess, 
        batch_size,
        test_size, 
        num_workers=4,
        pin_memory=True,        # 提升GPU传输效率  
        persistent_workers=True # 保持worker进程  
    ):
        
    dataset = WHURS19Dataset(
        data_path=data_path, 
        preprocess_func = preprocess
        )  
            
    # 划分数据集  
    train_data, test_data = train_test_split(
        dataset, 
        test_size=test_size, 
        random_state=42,
    ) 

    # 配置数据加载器  
    train_loader = DataLoader(  
        train_data,  
        batch_size=batch_size,  
        shuffle=True,  
        num_workers=num_workers,  
        pin_memory=pin_memory,  
        persistent_workers=persistent_workers,  
        prefetch_factor=2            # 提升数据预取  
    )  
    
    val_loader = DataLoader(  
        test_data,  
        batch_size=batch_size,  
        shuffle=False,               # 验证集不需要shuffle  
        num_workers=num_workers//2,  # 减少验证集workers  
        pin_memory=pin_memory,  
        persistent_workers=persistent_workers  
    ) 

    # 打印数据集统计信息  
    print(f"\n{' Dataset Info ':-^40}")  
    print(f"| {'Split':<15} | {'Samples':>8} |")  
    print(f"| {'-'*15} | {'-'*8} |")  
    print(f"| {'Training':<15} | {len(train_data):>8} |")  
    print(f"| {'Validation':<15} | {len(test_data):>8} |")  
    print(f"{'-'*40}\n")   

    return train_loader, val_loader  