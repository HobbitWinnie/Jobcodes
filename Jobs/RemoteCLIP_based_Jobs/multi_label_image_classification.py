import os  
import sys  
sys.path.append('/home/nw/Codes')  

import pandas as pd  
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  
from Models.RemoteCLIP_based_Classification.multi_label.factory import ClassifierFactory
from Loaders.MLRSNet_loader  import MLRSNetDataset

from utils.set_logging import setup_logging

def get_augmentation_transforms():  
    return transforms.Compose([  
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
    ])  

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
                image_path = os.path.join(images_dir, image_name)  
                
                if os.path.exists(image_path):  
                    data.append((image_path, labels))  
                else:  
                    print(f"Warning: Image {image_name} not found in {images_dir}")      
    return data  

if __name__ == "__main__":  
    
    checkpoint_path = "/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-L-14.pt"  
    model_name = 'ViT-L-14'

    # MLRSNetDataset
    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_512x512_dataset'
    images_dir = os.path.join(DATASET_DIR, 'Images')  
    labels_dir = os.path.join(DATASET_DIR, 'Labels')   
     
    """设置日志配置"""
    setup_logging()

    # 加载数据  
    data = load_MLRSNet_data(images_dir, labels_dir)  

    # 划分数据集  
    train_data, test_data = train_test_split(data, test_size=0.6, random_state=42) 

    # 初始化模型  
    num_labels = 17
    classifier = ClassifierFactory.create(  
        classifier_type='svm',  
        ckpt_path=checkpoint_path,  
        num_labels=num_labels,
        device_ids=[0,1,2,3]

    )  

    # 创建训练和测试数据集  
    augmentation_transforms = get_augmentation_transforms()
    preprocess_func= transforms.Compose([augmentation_transforms, classifier.preprocess_func])
    train_dataset = MLRSNetDataset(train_data, preprocess_func)  
    test_dataset = MLRSNetDataset(test_data, classifier.preprocess_func)  

    # 打印数据集的样本数量  
    print(f"Training dataset size: {len(train_dataset)}")  
    print(f"Testing dataset size: {len(test_dataset)}")  

    train_loader  = DataLoader(train_dataset, batch_size=192, num_workers=42, shuffle=True)  
    test_loader  = DataLoader(test_dataset, batch_size=192, num_workers=42, shuffle=True)  
   
    # 训练模型（FC）
    # classifier.train(train_loader, test_loader, num_epochs=10000)  

    # 训练模型（SVM, KNN）
    classifier.train(train_loader, num_epochs=10000)  
    metrics = classifier.evaluate(test_loader)  
    print(f"Test F1: {metrics['f1']:.4f}") 

    # 批量分类  
    classifier.classify_images('input_images', 'predictions.csv')  