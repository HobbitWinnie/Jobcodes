import os  
import pandas as pd  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  

from nw.Codes.Loaders.MultiLbel_loader import MultiLabelDataset  
from data_loader.MLRSNet_loader import MLRSNetDataset
from PIL import Image  
import pandas as pd  
from sklearn.metrics import f1_score  
import time  

from Model.ResNet import ResNetMultiLabelClassifier  
from Model.DenseNet201 import DenseNet201MultiLabelClassifier
from Model.VGG16 import VGG16MultiLabelClassifier
from Model.InceptionV3 import InceptionV3MultiLabelClassifier


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

def train_model(self, train_dataloader, val_dataloader, num_epochs=10):  
    self.model.train()  
    for epoch in range(num_epochs):  
        start_time = time.time()  # Start time for the epoch  
        running_loss = 0.0  
        for inputs, labels in train_dataloader:  
            inputs, labels = inputs.to(self.device), labels.to(self.device)  
            self.optimizer.zero_grad()  
            outputs = self.model(inputs)  
            loss = self.criterion(outputs, labels)  
            loss.backward()  
            self.optimizer.step()  
            running_loss += loss.item()  
    
        end_time = time.time()  # End time for the epoch  
        epoch_duration = end_time - start_time  # Calculate the duration  

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}, Time: {epoch_duration:.2f} seconds')  
        
        # Evaluate on validation set every 5 epochs  
        if (epoch + 1) % 5 == 0:  
            self.evaluate(val_dataloader)   
            self.model.train()  

def evaluate(self, dataloader, threshold=0.5):  
    self.model.eval()  
    all_labels = []  
    all_predictions = []  
    with torch.no_grad():  
        for inputs, labels in dataloader:  
            inputs, labels = inputs.to(self.device), labels.to(self.device)  
            outputs = self.model(inputs)  
            probabilities = torch.sigmoid(outputs).cpu().numpy()  
            predictions = (probabilities > threshold).astype(int)  
            
            all_labels.extend(labels.cpu().numpy())  
            all_predictions.extend(predictions)  

    # Calculate F1 score  
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)  
    print(f'F1 Score: {f1}')  
    return f1  

def classify_image(self, image_path):  
    try:  
        image = Image.open(image_path).convert("RGB")  
    except Exception as e:  
        print(f"Error opening image {image_path}: {e}")  
        return None  

    image_tensor = self.preprocess_func(image).unsqueeze(0).to(self.device)  
    self.model.eval()  
    with torch.no_grad():  
        output = self.model(image_tensor)  
        probabilities = torch.sigmoid(output).squeeze().cpu().numpy()  
    return probabilities  

def classify_images_in_folder(self, folder_path, output_csv):  
    results = []  
    for img_name in os.listdir(folder_path):  
        img_path = os.path.join(folder_path, img_name)  
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
            probabilities = self.classify_image(img_path)  
            if probabilities is not None:  
                results.append({"filename": img_name, "probabilities": probabilities})  

    df = pd.DataFrame(results)  
    df.to_csv(output_csv, index=False)  
    print(f"Results saved to `{output_csv}`")

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