import os  
import pandas as pd  
from torch.utils.data import DataLoader  
from data_loader import MultiLabelDataset  
from model import MultiLabelClassifierPro  


#单卡batch_size = 64，并行*显卡数量
def get_dataloaders(image_dir, label_file,  preprocess_func, batch_size=192, file_extension='.png'):  
    dataset = MultiLabelDataset(image_dir, label_file, preprocess_func, file_extension=file_extension)  
    # num_workers 的值可以设置为系统 CPU 核心数量的 2 到 4 倍，可以直接设置为与逻辑处理器数量相同
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=42, shuffle=True)  
    
    return train_loader 


if __name__ == "__main__":  

    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel'
    image_dir = os.path.join(DATASET_DIR, 'image_tianji')  
    label_file = os.path.join(DATASET_DIR, 'multilabel_all.csv')
    
    checkpoint_path = "/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-RN50.pt"  
    model_name = 'RN50'
    
    val_image_dir = image_dir 
    val_label_file = os.path.join(DATASET_DIR, 'multilabel_test.csv')  
    
    # test_image_dir = "path/to/test/images"  
    # output_csv = "classification_results.csv"  

    num_labels = len(pd.read_csv(label_file).columns) - 1  

    # 初始化模型  
    classifier = MultiLabelClassifierPro(num_labels, checkpoint_path, model_name)  

    # 获取数据加载器  
    train_loader = get_dataloaders(image_dir, label_file, classifier.preprocess_func)  
    
    # 验证集数据加载器  
    val_loader = get_dataloaders(val_image_dir, val_label_file, classifier.preprocess_func)  

    # 训练模型  
    classifier.train_model(train_loader, train_loader, num_epochs=50)  
    

    # 对测试集图像进行分类并保存结果  
    # classifier.classify_images_from_folder(test_image_dir, output_csv)  