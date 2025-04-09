import os  
from torch.utils.data import Dataset  
from PIL import Image  
import torch  

class MultiLabelDataset(Dataset):  
    def __init__(self, data_path, preprocess_func):  
        self.data_path = data_path  
        self.preprocess_func = preprocess_func  
        self.label_dict = self._generate_label_dict(data_path)  
        self.data = self._load_data(data_path)  
        
    def _generate_label_dict(self, data_path):  
        labels = set()  
        for folder_name in os.listdir(data_path):  
            folder_path = os.path.join(data_path, folder_name)  
            if os.path.isdir(folder_path):  
                labels.add(folder_name)  
        
        label_dict = {label: idx for idx, label in enumerate(sorted(labels))}  
        return label_dict  

    def _load_data(self, data_path):  
        data = []  
        for folder_name in os.listdir(data_path):  
            folder_path = os.path.join(data_path, folder_name)  
            if os.path.isdir(folder_path):  
                for file_name in os.listdir(folder_path):  
                    file_path = os.path.join(folder_path, file_name)  
                    if file_name.endswith(('.jpg', '.jpeg', '.png')):  
                        labels = [self.label_dict[folder_name]]  # Use the whole folder name as the label  
                        data.append((file_path, labels))  
        return data  

    def __len__(self):  
        return len(self.data)  

    def __getitem__(self, idx):  
        image_path, labels = self.data[idx]  
        image = Image.open(image_path).convert('RGB')  
        if self.preprocess_func:  
            image = self.preprocess_func(image)  
        
        target = torch.zeros(len(self.label_dict))  
        target[labels] = 1  
        return image, target, image_path  