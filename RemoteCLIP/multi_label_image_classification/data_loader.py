import os  
import pandas as pd  
import torch  
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
from PIL import Image  

class MultiLabelDataset(Dataset):  
    def __init__(self, image_dir, label_file, transform=None):  
        self.image_dir = image_dir  
        self.labels = pd.read_csv(label_file)  
        self.transform = transform  

    def __len__(self):  
        return len(self.labels)  

    def __getitem__(self, idx):  
        img_name = os.path.join(self.image_dir, self.labels.iloc[idx, 0])  
        image = Image.open(img_name).convert('RGB')  
        label = self.labels.iloc[idx, 1:].values.astype('float')  

        if self.transform:  
            image = self.transform(image)  

        return image, torch.tensor(label, dtype=torch.float32)   