import os  
import torch  
from torch.utils.data import Dataset  
from PIL import Image  
import pandas as pd  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  

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
    
def get_dataloaders(image_dir, label_file, preprocess_func, batch_size=192, file_extension='.png'): 
    # 读取原始 CSV 文件  
    data = pd.read_csv(label_file)
    # Create datasets and dataloaders  
    train_labels, test_labels = train_test_split(data, test_size=0.33, random_state=42)  
    train_dataset = MultiLabelDataset(image_dir, train_labels, preprocess_func, file_extension=file_extension)  
    test_dataset = MultiLabelDataset(image_dir, test_labels, preprocess_func, file_extension=file_extension)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=42, shuffle=True)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=42, shuffle=True)  

    return train_loader, test_loader  