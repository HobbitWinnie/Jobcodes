import os  
import pandas as pd  
from torchvision import transforms  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  


from data_loader import MultiLabelDataset  
from ResNet_classifier import ResNetMultiLabelClassifier  
from DenseNet201_classifier import DenseNet201MultiLabelClassifier
from VGG16_classifier import VGG16MultiLabelClassifier
from InceptionV3_classifier import InceptionV3MultiLabelClassifier


def get_dataloaders(image_dir, label_file, preprocess_func, batch_size=192, file_extension='.png'): 
    # 读取原始 CSV 文件  
    data = pd.read_csv(label_file)
    # Create datasets and dataloaders  
    train_labels, test_labels = train_test_split(data, test_size=0.2, random_state=42)  

    train_dataset = MultiLabelDataset(image_dir, train_labels, preprocess_func, file_extension=file_extension)  
    test_dataset = MultiLabelDataset(image_dir, test_labels, preprocess_func, file_extension=file_extension)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=42, shuffle=True)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=42, shuffle=True)  

    return train_loader, test_loader  


if __name__ == "__main__":  

    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel'
    image_dir = os.path.join(DATASET_DIR, 'image_tianji')  
    label_file = os.path.join(DATASET_DIR, 'multilabel_all.csv')   
     
    num_labels = len(pd.read_csv(label_file).columns) - 1  
   
    # 初始化模型  
    classifier = DenseNet201MultiLabelClassifier(num_classes=num_labels)  
    # classifier = DenseNet201MultiLabelClassifier(num_classes=num_labels)  

    # 获取数据加载器  
    train_dataloader, test_loader = get_dataloaders(image_dir, label_file, classifier.preprocess_func)  
    
    # 训练模型  
    classifier.train_model(train_dataloader, test_loader, num_epochs=100)  
