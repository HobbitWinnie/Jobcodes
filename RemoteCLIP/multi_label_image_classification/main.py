import os  
import pandas as pd  
from torchvision import transforms  
from torch.utils.data import DataLoader  
from data_loader import MultiLabelDataset  
from model import MultiLabelClassifierPro  


def get_dataloaders(image_dir, label_file,  preprocess_func, batch_size=64, file_extension='.png'):  
    dataset = MultiLabelDataset(image_dir, label_file, preprocess_func, file_extension=file_extension)  
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  
    return train_loader 


if __name__ == "__main__":  

    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel'
    image_dir = os.path.join(DATASET_DIR, 'image_tianji')  
    label_file = os.path.join(DATASET_DIR, 'multilabel_train.csv')
    
    checkpoint_path = "/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-RN50.pt"  
    model_name = 'RN50'
    
    val_image_dir = image_dir 
    val_label_file = os.path.join(DATASET_DIR, 'multilabel_test.csv')  
    
    # test_image_dir = "path/to/test/images"  
    # output_csv = "classification_results.csv"  

 
    # 初始化模型  
    classifier = MultiLabelClassifierPro(checkpoint_path, model_name)  

    # 获取数据加载器  
    train_loader = get_dataloaders(image_dir, label_file, classifier.preprocess_func)  
    num_labels = len(pd.read_csv(label_file).columns) - 1  
     # 验证集数据加载器  
    val_loader = get_dataloaders(val_image_dir, val_label_file, classifier.preprocess_func)  

    # 训练模型  
    classifier.train_model(train_loader, val_loader, num_labels, num_epochs=100, lr=1e-4, criterion=None)  
    

    # 对测试集图像进行分类并保存结果  
    # classifier.classify_images_from_folder(test_image_dir, output_csv)  