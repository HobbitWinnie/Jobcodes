
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from model import SSDGLNet
from loss import WeightedCrossEntropyLoss
from data_loader import create_dataloaders
from config import get_config


# 学习率调度器（Poly衰减策略）  
def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=1000, power=0.8):  
    if iter % lr_decay_iter or iter > max_iter:  
        return optimizer  
    lr = init_lr * (1 - iter / max_iter) ** power  
    for param_group in optimizer.param_groups:  
        param_group['lr'] = lr  
    return lr  

# 模型训练函数  
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):  
    model.train()  
    total_iters = len(train_loader) * num_epochs  
    current_iter = 0  
    for epoch in range(num_epochs):  
        for data, targets in train_loader:  
            data, targets = data.to(device), targets.to(device)  
            optimizer.zero_grad()  
            outputs = model(data)  
            outputs = outputs.view(-1, outputs.size(-1))  # 调整尺寸以匹配损失函数输入  
            targets = targets.view(-1)  
            loss = criterion(outputs, targets)  
            loss.backward()  
            optimizer.step()  

            # 更新学习率  
            current_iter += 1  
            poly_lr_scheduler(optimizer, init_lr=optimizer.param_groups[0]['initial_lr'],  
                              iter=current_iter, max_iter=1000, power=0.8)  

# 模型评估函数  
def evaluate_model(model, test_loader, device):  
    model.eval()  
    correct = 0  
    total = 0  
    with torch.no_grad():  
        for data, targets in test_loader:  
            data, targets = data.to(device), targets.to(device)  
            outputs = model(data)  
            outputs = outputs.view(-1, outputs.size(-1))  
            targets = targets.view(-1)  
            _, predicted = torch.max(outputs.data, 1)  
            total += targets.size(0)  
            correct += (predicted == targets).sum().item()  
    accuracy = 100 * correct / total  
    print('测试准确率: {:.2f}%'.format(accuracy))  

def train_loop(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    # 初始化模型、损失函数和优化器  
    model = SSDGLNet(num_classes=num_classes).to(device)  

    # 计算每个类别的权重  
    class_counts = np.bincount(labels.flatten())  
    class_weights = 1. / (class_counts + 1e-6)  
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        image_dir= Path(config['paths']['data']['image_dir']),
        labels_dir=Path(config['paths']['data']['label_dir']),
        batch_size=config['training']['batch_size'],
        train_ratio=config['dataset']['train_val_split'],
        num_workers=config['dataset']['num_workers'],
    )

    criterion = WeightedCrossEntropyLoss(weights=class_weights)  
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)  
    for param_group in optimizer.param_groups:  
        param_group['initial_lr'] = learning_rate  # 保存初始学习率  

    # 训练和评估模型  
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)  
    evaluate_model(model, val_loader, device)  

    # 保存模型  
    torch.save(model.state_dict(), 'ssdgl_model.pth')  


# 主函数
def main():

    config = get_config()

    train_loop(config)


if __name__ == '__main__':
    main()

