import os  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms, models  
from PIL import Image  
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score  
import numpy as np  

# 自定义数据集类  
class CustomDataset(Dataset):  
    def __init__(self, image_paths, labels, transform=None):  
        self.image_paths = image_paths  
        self.labels = labels  
        self.transform = transform  

    def __len__(self):  
        return len(self.image_paths)  

    def __getitem__(self, idx):  
        image = Image.open(self.image_paths[idx]).convert('RGB')  
        label = self.labels[idx]  
        if self.transform:  
            image = self.transform(image)  
        return image, torch.tensor(label, dtype=torch.float32)  

# 数据准备  
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]  # 替换为你的图像路径列表  
labels = [[1, 0, 1], [0, 1, 0], ...]  # 替换为你的标签列表  

transform = transforms.Compose([  
    transforms.Resize((224, 224)),  
    transforms.ToTensor()  
])  

train_dataset = CustomDataset(image_paths, labels, transform=transform)  
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  

# 模型定义  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = models.resnet18(pretrained=True)  
num_labels = len(labels[0])  # 标签的数量  
model.fc = nn.Linear(model.fc.in_features, num_labels)  
model = model.to(device)  

# 损失函数和优化器  
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  

# 训练过程  
num_epochs = 10  

for epoch in range(num_epochs):  
    model.train()  
    total_loss = 0.0  
    for images, labels in train_loader:  
        images, labels = images.to(device), labels.to(device)  
        
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
        
        total_loss += loss.item()  
    
    avg_loss = total_loss / len(train_loader)  
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")  

# 评估函数  
def evaluate_model(model, dataloader):  
    model.eval()  
    all_targets = []  
    all_predictions = []  

    with torch.no_grad():  
        for images, labels in dataloader:  
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)  
            all_targets.append(labels.cpu().numpy())  
            all_predictions.append(outputs.sigmoid().cpu().numpy())  

    all_targets = np.concatenate(all_targets, axis=0)  
    all_predictions = np.concatenate(all_predictions, axis=0)  

    threshold = 0.5  
    all_predictions_bin = (all_predictions > threshold).astype(int)  
    f1 = f1_score(all_targets, all_predictions_bin, average='weighted')  
    average_precision = average_precision_score(all_targets, all_predictions, average='weighted')  
    roc_auc = roc_auc_score(all_targets, all_predictions, average='weighted')  

    print(f"F1 Score: {f1}")  
    print(f"Average Precision: {average_precision}")  
    print(f"ROC-AUC: {roc_auc}")  

    return f1, average_precision, roc_auc  

# 假设你有一个验证集的数据加载器  
val_image_paths = ["path/to/val_image1.jpg", "path/to/val_image2.jpg", ...]  # 替换为你的验证图像路径列表  
val_labels = [[1, 0, 1], [0, 1, 0], ...]  # 替换为你的验证标签列表  

val_dataset = CustomDataset(val_image_paths, val_labels, transform=transform)  
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  

# 评估模型  
evaluate_model(model, val_loader)