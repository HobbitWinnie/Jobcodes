import os  
import sys  
sys.path.append('/home/nw/Codes')  

import torchvision.transforms as transforms  
from Models.RemoteCLIP_based_Classification.multi_label.factory import ClassifierFactory
from Loaders.MLRSNet_loader import get_dataloaders

from utils.set_logging import setup_logging

def get_augmentation_transforms():  
    return transforms.Compose([  
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
    ])  


if __name__ == "__main__":  
    
    checkpoint_path = "/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-L-14.pt"  
    model_name = 'ViT-L-14'

    # MLRSNetDataset
    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_512x512_dataset'
    images_dir = os.path.join(DATASET_DIR, 'Images')  
    labels_dir = os.path.join(DATASET_DIR, 'Labels')   
     
    """设置日志配置"""
    setup_logging()

    # 初始化模型  
    num_labels = 17
    classifier = ClassifierFactory.create(  
        classifier_type='fc',  
        ckpt_path=checkpoint_path,  
        num_labels=num_labels,
        device_ids=[2,3]
    )  

    # 创建训练和测试数据集  
    augmentation_transforms = get_augmentation_transforms()
    preprocess_func= transforms.Compose([augmentation_transforms, classifier.preprocess_func])
    train_loader, test_loader = get_dataloaders(
        images_dir=images_dir,
        labels_dir=labels_dir,
        preprocess=preprocess_func,
        batch_size=192,
        test_size=0.6
    )

    # 训练模型
    classifier.train(train_loader, test_loader, num_epochs=10000)  

    # 批量分类  
    # classifier.classify_images('input_images', 'predictions.csv')  