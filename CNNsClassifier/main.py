import os  
import pandas as pd  
from torchvision import transforms  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  

from data_loader import MultiLabelDataset  
from MLRSNet_loader import MLRSNetDataset

from ResNet_classifier import ResNetMultiLabelClassifier  
from DenseNet201_classifier import DenseNet201MultiLabelClassifier
from VGG16_classifier import VGG16MultiLabelClassifier
from InceptionV3_classifier import InceptionV3MultiLabelClassifier


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

def load_MLRSNet_data(images_dir, labels_dir):  
    """加载所有图像和标签数据"""  
    data = []  
    for label_file in os.listdir(labels_dir):  
        if label_file.endswith('.csv'):  
            label_path = os.path.join(labels_dir, label_file)  
            label_data = pd.read_csv(label_path)  
            category = label_file.replace('.csv', '')  
            image_folder = os.path.join(images_dir, category)  
            
            for _, row in label_data.iterrows():  
                image_name = row.iloc[0]  
                labels = row.iloc[1:].values.astype('float')  
                image_path = os.path.join(image_folder, image_name)  
                
                if os.path.exists(image_path):  
                    data.append((image_path, labels))  
                else:  
                    print(f"Warning: Image {image_name} not found in {image_folder}")      
    return data  


if __name__ == "__main__":  

    # DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel'
    # image_dir = os.path.join(DATASET_DIR, 'image_tianji')  
    # label_file = os.path.join(DATASET_DIR, 'multilabel_all.csv')   
     
    # num_labels = len(pd.read_csv(label_file).columns) - 1  
   
    # # 初始化模型  
    # classifier = ResNetMultiLabelClassifier(num_classes=num_labels)  

    # # 获取数据加载器  
    # train_loader, test_loader = get_dataloaders(image_dir, label_file, classifier.preprocess_func)  
    
    # # 训练模型  
    # classifier.train_model(train_dataloader, test_loader, num_epochs=100)  


    # MLRSNetDataset
    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/MLRSNet_dataset'
    images_dir = os.path.join(DATASET_DIR, 'Images')  
    labels_dir = os.path.join(DATASET_DIR, 'Labels')   
     
    # 加载数据  
    data = load_MLRSNet_data(images_dir, labels_dir)  

    # 划分数据集  
    train_data, test_data = train_test_split(data, test_size=0.9, random_state=42) 

    # 初始化模型  
    num_labels = 60
    classifier = ResNetMultiLabelClassifier(num_classes=num_labels)  

    # 创建训练和测试数据集  
    train_dataset = MLRSNetDataset(train_data, classifier.preprocess_func)  
    test_dataset = MLRSNetDataset(test_data, classifier.preprocess_func)  

    # 打印数据集的样本数量  
    print(f"Training dataset size: {len(train_dataset)}")  
    print(f"Testing dataset size: {len(test_dataset)}")  

    train_loader  = DataLoader(train_dataset, batch_size=192, num_workers=42, shuffle=True)  
    test_loader  = DataLoader(test_dataset, batch_size=192, num_workers=42, shuffle=True)  
   
    # 训练模型  
    classifier.train_model(train_loader, test_loader, num_epochs=100)  