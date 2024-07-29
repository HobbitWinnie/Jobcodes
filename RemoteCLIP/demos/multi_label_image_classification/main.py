import os  
import pandas as pd  
from torchvision import transforms  
from torch.utils.data import DataLoader  
from data_loader import MultiLabelDataset  
from model import MultiLabelClassifier  

def get_dataloaders(image_dir, label_file, batch_size=32):  
    transform = transforms.Compose([  
        transforms.Resize((224, 224)),  
        transforms.ToTensor()  
    ])  

    dataset = MultiLabelDataset(image_dir, label_file, transform=transform)  
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  

    return train_loader 


if __name__ == "__main__":  
    ucm_image_dir = "path/to/ucm/images"  
    ucm_label_file = "path/to/ucm/labels.csv"  
    
    checkpoint_path = "path/to/checkpoint.pth"  
    
    val_image_dir = "path/to/val/images"  
    val_label_file = "path/to/val/labels.csv"  
    
    test_image_dir = "path/to/test/images"  
    output_csv = "classification_results.csv"  

    transform = transforms.Compose([  
        transforms.Resize((224, 224)),  
        transforms.ToTensor()  
    ])  

    # 获取数据加载器  
    train_loader = get_dataloaders(ucm_image_dir, ucm_label_file)  
    num_labels = len(pd.read_csv(ucm_label_file).columns) - 1  
 
    # 初始化模型  
    classifier = MultiLabelClassifier(num_labels, checkpoint_path)  

    # 训练模型  
    classifier.train_model(train_loader, num_labels=num_labels, num_epochs=10, lr=1e-4, loss_type='bce')  

    # 验证集数据加载器  
    val_dataset = MultiLabelDataset(val_image_dir, val_label_file, transform=transform)  
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  

    # 评估模型  
    classifier.evaluate_model(val_loader)  

    # 对测试集图像进行分类并保存结果  
    classifier.classify_images_from_folder(test_image_dir, output_csv)  