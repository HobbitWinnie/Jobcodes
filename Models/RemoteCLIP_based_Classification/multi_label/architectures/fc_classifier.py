
import torch
import numpy as np
from torch import nn, optim
from ..core.base import BaseCLIPClassifier

class FCClassifier(BaseCLIPClassifier):
    """全连接分类器实现"""
    
    def __init__(self, ckpt_path: str, num_labels: int, **kwargs):
        super().__init__(ckpt_path, **kwargs)
        self.num_labels = num_labels
        self._build_model()
        self._init_optimizer()

    def _build_model(self):
        """构建分类头"""
        model = self.model.module if hasattr(self.model, 'module') else self.model  
        self.classifier = nn.Sequential(
            nn.Linear(model.visual.output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_labels)
        ).to(self.main_device).float()

    def _init_optimizer(self):
        """初始化优化器"""
        self.criterion = nn.BCEWithLogitsLoss().to(self.main_device)  
        self.optimizer = optim.AdamW(self.classifier.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=0.01,
            total_steps=100
        )

    def _train_impl(self, features, labels, num_epochs=50):
        """具体训练实现"""
        features = torch.from_numpy(features).float().to(self.main_device)
        labels = torch.from_numpy(labels).float().to(self.main_device)

        scaler = torch.cuda.amp.GradScaler(enabled=self.main_device.startswith('cuda'))  
        self.classifier.train()

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.main_device.startswith('cuda')):  
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
        
            # 使用scaler进行梯度缩放  
            scaler.scale(loss).backward()  
            scaler.step(self.optimizer)  
            scaler.update()              
            self.scheduler.step() 

            if (epoch+1) % 5 == 0:
                self.logger.info(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    def _get_predictions(self, data_loader):
        """获取预测结果"""
        self.classifier.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in data_loader:
                # 特征提取使用基类方法  
                features = self._get_features(images.to(self.main_device))  

                # 转换到主设备  
                features_tensor = torch.from_numpy(features).to(self.main_device)  

                with torch.cuda.amp.autocast(enabled=self.main_device.startswith('cuda')):  
                    outputs = self.classifier(features_tensor)

                all_preds.append(outputs.sigmoid().cpu().numpy())
                all_labels.append(labels.numpy())

        return np.concatenate(all_labels), np.concatenate(all_preds)

    def _format_prediction(self, features):
        """格式化预测结果"""
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).to(self.main_device)  
            outputs = self.classifier(features_tensor)
            probs = outputs.sigmoid().cpu().numpy()[0]
        return {f'label_{i}': float(p) for i, p in enumerate(probs)}
