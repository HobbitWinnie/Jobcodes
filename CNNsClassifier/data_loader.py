import os  
import pandas as pd  
import torch  
from torch.utils.data import Dataset  
from torchvision import transforms  
from PIL import Image  

class MultiLabelDataset(Dataset):  
    def __init__(self, image_dir, labels, preprocess_func, file_extension='.png'):  
        self.image_dir = image_dir  
        self.labels = labels
        self.file_extension = file_extension
        self.preprocess_func = preprocess_func

    def __len__(self):  
        return len(self.labels)  

    def __getitem__(self, idx):  
        img_name = os.path.join(self.image_dir, self.labels.iloc[idx, 0]) + self.file_extension
        image = Image.open(img_name).convert('RGB')  
        image_tensor = self.preprocess_func(image)
        label = self.labels.iloc[idx, 1:].values.astype('float')  
        
        return image_tensor, torch.tensor(label, dtype=torch.float32)   