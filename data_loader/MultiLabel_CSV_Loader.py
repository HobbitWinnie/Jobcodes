import torch  
from torch.utils.data import Dataset  
from PIL import Image  
import pandas as pd  

class MultiLabelCSVLoader(Dataset):  
    def __init__(self, csv_path, preprocess_func):  
        self.data = pd.read_csv(csv_path)  
        self.preprocess_func = preprocess_func  
        self.labels = self.data.columns[1:]  # Skip 'image_path' column  

    def __len__(self):  
        return len(self.data)  

    def __getitem__(self, idx):  
        image_path = self.data.iloc[idx, 0]  # Read full image path  
        image = Image.open(image_path).convert('RGB')  

        if self.preprocess_func:  
            image = self.preprocess_func(image)  

        labels = torch.tensor(self.data.iloc[idx, 1:].values.astype('float32'))  
        return image, labels, image_path