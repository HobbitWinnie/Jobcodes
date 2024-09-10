import os  
import sys  
from torch.utils.data import DataLoader  
import pandas as pd  
from sklearn.model_selection import train_test_split  

sys.path.append('/home/nw/Codes/data_loader')  
sys.path.append('/home/nw/Codes/RemoteCLIP/Image_Classification/src')  

from remote_clip_mlknn import RemoteCLIPClassifierMLKNN
from remote_clip_ranksvm import RemoteCLIPClassifierRankSVM
from MLRSNet_loader import MLRSNetDataset


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
    
    checkpoint_path = "/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-L-14.pt"  
    model_name = 'ViT-L-14'

    # MLRSNetDataset
    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/MLRSNet_dataset'
    images_dir = os.path.join(DATASET_DIR, 'Images')  
    labels_dir = os.path.join(DATASET_DIR, 'Labels')   
     
    # 加载数据  
    data = load_MLRSNet_data(images_dir, labels_dir)  

    # 划分数据集  
    train_data, test_data = train_test_split(data, test_size=0.5, random_state=42) 

    # 初始化模型  
    num_labels = 60
    # classifier = RemoteCLIPClassifierMLKNN(checkpoint_path, model_name)  
    
    classifier = RemoteCLIPClassifierRankSVM(checkpoint_path, model_name)  


    # 创建训练和测试数据集  
    train_dataset = MLRSNetDataset(train_data, classifier.preprocess_func)  
    test_dataset = MLRSNetDataset(test_data, classifier.preprocess_func)  

    # 打印数据集的样本数量  
    print(f"Training dataset size: {len(train_dataset)}")  
    print(f"Testing dataset size: {len(test_dataset)}")  

    train_loader  = DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)  
    test_loader  = DataLoader(test_dataset, batch_size=128, num_workers=12, shuffle=True)  
   
    # 训练模型   
    # classifier.fit_knn(train_loader)  
    classifier.train_model(train_loader)  
    classifier.evaluate(test_loader)