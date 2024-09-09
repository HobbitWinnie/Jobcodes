import os 
import sys
import pandas as pd  
from torch.utils.data import DataLoader  
from data_loader import MultiLabelDataset  
from model import MultiLabelClassifierPro  
from sklearn.model_selection import train_test_split  

sys.path.append('/home/nw/Codes/data_loader')  
from MLRSNet_loader import MLRSNetDataset


#单卡batch_size = 64，并行*显卡数量
def get_dataloaders(image_dir, label_file,  preprocess_func, batch_size=192, file_extension='.png'):  
    dataset = MultiLabelDataset(image_dir, label_file, preprocess_func, file_extension=file_extension)  
    # num_workers 的值可以设置为系统 CPU 核心数量的 2 到 4 倍，可以直接设置为与逻辑处理器数量相同
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=42, shuffle=True)  
    
    return train_loader 

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
    
    checkpoint_path = "/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-L-14.pt"  
    model_name = 'ViT-L-14'
    
    # val_image_dir = image_dir 
    # val_label_file = os.path.join(DATASET_DIR, 'multilabel_test.csv')  
    
    # # test_image_dir = "path/to/test/images"  
    # # output_csv = "classification_results.csv"  

    # num_labels = len(pd.read_csv(label_file).columns) - 1  

    # # 初始化模型  
    # classifier = MultiLabelClassifierPro(num_labels, checkpoint_path, model_name)  

    # # 获取数据加载器  
    # train_loader = get_dataloaders(image_dir, label_file, classifier.preprocess_func)  
    
    # # 验证集数据加载器  
    # val_loader = get_dataloaders(val_image_dir, val_label_file, classifier.preprocess_func)  

    # # 训练模型  
    # classifier.train_model(train_loader, train_loader, num_epochs=50)  
    



    # MLRSNetDataset
    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/MLRSNet_dataset'
    images_dir = os.path.join(DATASET_DIR, 'Images')  
    labels_dir = os.path.join(DATASET_DIR, 'Labels')   
     
    # 加载数据  
    data = load_MLRSNet_data(images_dir, labels_dir)  

    # 划分数据集  
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42) 

    # 初始化模型  
    num_labels = 60
    classifier = MultiLabelClassifierPro(num_labels, checkpoint_path, model_name)  

    # 创建训练和测试数据集  
    train_dataset = MLRSNetDataset(train_data, classifier.preprocess_func)  
    test_dataset = MLRSNetDataset(test_data, classifier.preprocess_func)  

    # 打印数据集的样本数量  
    print(f"Training dataset size: {len(train_dataset)}")  
    print(f"Testing dataset size: {len(test_dataset)}")  

    train_loader  = DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)  
    test_loader  = DataLoader(test_dataset, batch_size=128, num_workers=12, shuffle=True)  
   
    # 训练模型   
    classifier.train_model(train_loader, test_loader, num_epochs=100)  
    
    # # 对测试集图像进行分类并保存结果  
    # # classifier.classify_images_from_folder(test_image_dir, output_csv)  