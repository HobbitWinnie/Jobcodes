import os  
from pathlib import Path  
from PIL import Image  
from torch.utils.data import Dataset  

# 数据集类  
class MillionAIDDatasetLoader(Dataset):  
    def __init__(self, data_path, preprocess_func):  
        self.preprocess_func = preprocess_func  
        self.image_paths = []  
        self.labels = []  
        self.classes = set()  

        # 收集所有可能的标签  
        self._collect_classes(data_path)  

        self.classes = sorted(self.classes)  
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  

        # 打印收集到的标签  
        print("Collected classes (labels):", self.classes)  

        # 收集图像路径和对应标签  
        self._collect_image_paths_and_labels(data_path)  

    def _collect_classes(self, path):  
        for item in Path(path).rglob('*'):  
            if item.is_dir():  
                self.classes.add(item.name)  

    def _collect_image_paths_and_labels(self, path):  
        for item in Path(path).rglob('*.*'):  
            if item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:  
                parent_dir_name = item.parent.name  
                if parent_dir_name in self.class_to_idx:  
                    label_idx = self.class_to_idx[parent_dir_name]  
                    self.image_paths.append(item)  
                    self.labels.append(label_idx)  

    def __len__(self):  
        return len(self.image_paths)  

    def __getitem__(self, idx):  
        image_path = self.image_paths[idx]  
        label = self.labels[idx]  
        image = Image.open(image_path).convert('RGB')  
        image = self.preprocess_func(image)  

        return image, label, str(image_path)  