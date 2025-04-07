
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn.metrics import accuracy_score

# 模拟数据集加载器（需要根据 WHU-Hi 数据集实际格式调整）
class HyperspectralDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data: 高光谱影像数据，形状为 [N, C, H, W]，N 是样本数，C 是光谱通道数
            labels: 对应的标签，形状为 [N, H, W]
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Additive Angular Margin Loss (AAM Loss)
class AAMLoss(nn.Module):
    def __init__(self, margin=0.5, scale=30):
        super(AAMLoss, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits, labels):
        # Normalize logits
        logits = F.normalize(logits, dim=1)
        # Add margin to the correct class
        one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
        logits_with_margin = logits - one_hot * self.margin
        # Scale logits
        scaled_logits = logits_with_margin * self.scale
        return F.cross_entropy(scaled_logits, labels)


# 模型训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Args:
        model: S3ANet 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数（如 AAMLoss）
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备（'cuda' 或 'cpu'）
    """
    model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)  # 输出形状为 [B, num_classes, H, W]
            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1))  # 展平为 [B*H*W, num_classes]
            labels = labels.view(-1)  # 展平为 [B*H*W]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1))
                labels = labels.view(-1)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_s3anet_model.pth")
            print("Best model saved!")

    print("Training complete. Best Val Acc: {:.4f}".format(best_val_acc))


# 主函数
if __name__ == "__main__":
    # 数据准备（需要根据实际数据集调整）
    # 模拟高光谱数据 [N, C, H, W] 和标签 [N, H, W]
    num_samples = 100  # 样本数
    num_channels = 128  # 光谱通道数
    height, width = 64, 64  # 图像尺寸
    num_classes = 5  # 类别数

    # 随机生成数据和标签
    data = torch.rand(num_samples, num_channels, height, width)
    labels = torch.randint(0, num_classes, (num_samples, height, width))

    # 划分训练集和验证集
    train_data, val_data = data[:80], data[80:]
    train_labels, val_labels = labels[:80], labels[80:]

    train_dataset = HyperspectralDataset(train_data, train_labels)
    val_dataset = HyperspectralDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 模型初始化
    from model import S3ANet  # 假设模型代码保存在 s3anet_model.py 文件中
    model = S3ANet(in_channels=num_channels, num_classes=num_classes)

    # 损失函数和优化器
    criterion = AAMLoss(margin=0.5, scale=30)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.001)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)

