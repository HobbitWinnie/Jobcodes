import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import datasets, transforms, models  

# 数据准备  
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)  

# 模型定义  
model = models.resnet18(pretrained=True)  
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10有10个类别  
model = model.to(device)  

# 损失函数和优化器  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  

# 训练过程  
for epoch in range(num_epochs):  
    model.train()  
    for images, labels in train_loader:  
        images, labels = images.to(device), labels.to(device)  
        
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()