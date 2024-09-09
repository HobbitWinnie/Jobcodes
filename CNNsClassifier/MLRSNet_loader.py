import os  
import torch  
from torch.utils.data import Dataset, DataLoader  
from PIL import Image  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from torchvision import transforms  

class MLRSNetDataset(Dataset):  
    def __init__(self, data, preprocess_func):  
        """  
        初始化MultiLabelDataset类。  
        
        :param data: 包含图像路径和标签的列表  
        :param preprocess_func: 图像预处理函数  
        """  
        self.data = data  
        self.preprocess_func = preprocess_func  

    def __len__(self):  
        """返回数据集中样本的数量"""  
        return len(self.data)  

    def __getitem__(self, idx):  
        """  
        根据索引获取样本。  
        
        :param idx: 数据集中样本的索引  
        :return: 预处理后的图像张量和标签张量  
        """  
        img_path, labels = self.data[idx]  
        image = Image.open(img_path).convert('RGB')  
        image_tensor = self.preprocess_func(image)  
        
        return image_tensor, torch.tensor(labels, dtype=torch.float32)  