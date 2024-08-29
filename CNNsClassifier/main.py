import os  
import pandas as pd  
from torchvision import transforms  
from torch.utils.data import DataLoader  

from data_loader import MultiLabelDataset  
from ResNet_classifier import ResNetMultiLabelClassifier  
from DenseNet201_classifier import DenseNet201MultiLabelClassifier
from VGG16_classifier import VGG16MultiLabelClassifier
from InceptionV3_classifier import InceptionV3MultiLabelClassifier


#单卡batch_size = 64，并行*显卡数量
def get_dataloaders(image_dir, label_file, preprocess_func, batch_size=192, file_extension='.png'):  
    dataset = MultiLabelDataset(image_dir, label_file, preprocess_func, file_extension=file_extension)  
    # num_workers 的值可以设置为系统 CPU 核心数量的 2 到 4 倍，可以直接设置为与逻辑处理器数量相同
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=42, shuffle=True)  
    
    return train_loader 


if __name__ == "__main__":  

    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel'
    image_dir = os.path.join(DATASET_DIR, 'image_tianji')  
    label_file = os.path.join(DATASET_DIR, 'multilabel_all.csv')   
    val_label_file = os.path.join(DATASET_DIR, 'multilabel_test.csv')  
     
    num_labels = len(pd.read_csv(label_file).columns) - 1  
   
    # 初始化模型  
    classifier = VGG16MultiLabelClassifier(num_classes=num_labels)  
    # classifier = DenseNet201MultiLabelClassifier(num_classes=num_labels)  

    # 获取数据加载器  
    train_dataloader = get_dataloaders(image_dir, label_file, classifier.preprocess_func)  
    val_dataloader = get_dataloaders(image_dir, val_label_file, classifier.preprocess_func)  
    
    # 训练模型  
    classifier.train_model(train_dataloader, val_dataloader, num_epochs=50)  
