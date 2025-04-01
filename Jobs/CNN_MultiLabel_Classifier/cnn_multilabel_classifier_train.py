import sys  
sys.path.append('/home/nw/Codes')  

import os  
import time  
import pandas as pd  
import torch  
import torch.nn as nn  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import f1_score  

from Loaders.MultiLbel_loader import MultiLabelDataset  
from Loaders.MLRSNet_loader import MLRSNetDataset

from Models.CNN_MultiLabel_Classifier.model_factory import create_model


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


def train_model(model, train_loader, val_loader, num_epochs=10):  
    criterion = nn.BCEWithLogitsLoss()  
    best_f1 = 0  
    model.train()  
    
    for epoch in range(num_epochs):  
        epoch_loss = 0  
        start_time = time.time()  
        
        for inputs, labels in train_loader:  
            inputs = inputs.to(model.device)  
            labels = labels.float().to(model.device)  
            
            model.optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            model.optimizer.step()  
            
            epoch_loss += loss.item()  

        # 验证和保存最佳模型  
        val_f1 = evaluate(model, val_loader)  
        if val_f1 > best_f1:  
            torch.save(model.state_dict(), "best_model.pth")  
            best_f1 = val_f1  
            
        print(f"Epoch {epoch+1}/{num_epochs} | "  
              f"Loss: {epoch_loss/len(train_loader):.4f} | "  
              f"Val F1: {val_f1:.4f} | "  
              f"Time: {time.time()-start_time:.1f}s")  

def evaluate(model, dataloader, threshold=0.5):  
    model.eval()  
    all_preds = []  
    all_labels = []  
    
    with torch.no_grad():  
        for inputs, labels in dataloader:  
            inputs = inputs.to(model.device)  
            outputs = model(inputs)  
            
            probs = torch.sigmoid(outputs).cpu()  
            preds = (probs > threshold).int()  
            
            all_preds.extend(preds.numpy())  
            all_labels.extend(labels.int().numpy())  
    
    return f1_score(all_labels, all_preds, average="macro")  


if __name__ == "__main__":  

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
    model = create_model('resnet101', num_labels, device='cuda:0', multi_gpu=False)  

    # 创建训练和测试数据集  
    train_dataset = MLRSNetDataset(train_data, model.preprocess)  
    test_dataset = MLRSNetDataset(test_data, model.preprocess)  

    # 打印数据集的样本数量  
    print(f"Training dataset size: {len(train_dataset)}")  
    print(f"Testing dataset size: {len(test_dataset)}")  

    train_loader = DataLoader(train_dataset, batch_size=192, num_workers=42, shuffle=True)  
    test_loader = DataLoader(test_dataset, batch_size=192, num_workers=42, shuffle=True)  
   
    # 训练模型  
    train_model(model, train_loader, test_loader, num_epochs=100)  