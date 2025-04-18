import torch  
import numpy as np  
from torch import nn, optim  
from torch.optim.lr_scheduler import OneCycleLR  
from datetime import datetime  
import time  
from ..core.base import BaseCLIPClassifier  


class FullModel(nn.Module):  
    def __init__(self, clip_model, classifier):  
        super().__init__()  
        self.clip_model = clip_model  
        self.classifier = classifier  

    def forward(self, x):  
        x = self.clip_model.encode_image(x)  
        x = x / x.norm(dim=-1, keepdim=True)  
        return self.classifier(x)  


class FCClassifier(BaseCLIPClassifier):  
    """全连接分类器实现"""  

    def __init__(self, ckpt_path: str, num_labels: int, **kwargs):  
        super().__init__(ckpt_path, **kwargs)  
        self.num_labels = num_labels  
        self._build_model()  
        self._init_optimizer()  

    def _build_model(self):  
        clip_model = self.clip_model.module if hasattr(self.clip_model, 'module') else self.clip_model  
        self.classifier = nn.Sequential(  
            nn.Linear(clip_model.visual.output_dim, 512),  
            nn.BatchNorm1d(512),  
            nn.ReLU(),  
            nn.Dropout(0.5),  
            nn.Linear(512, self.num_labels)  
        ).to(self.main_device).float()  

        self.full_model = FullModel(clip_model, self.classifier).to(self.main_device).float()  

        if len(self.device_ids) > 1:  
            self.full_model = torch.nn.DataParallel(self.full_model, device_ids=self.device_ids)  
            self.logger.info(f"模型已包裹 DataParallel，使用GPU: {self.device_ids}")  

    def _init_optimizer(self):  
        self.criterion = nn.BCEWithLogitsLoss().to(self.main_device)  
        self.optimizer = optim.AdamW(self.classifier.parameters(), lr=0.001)  
        # scheduler 不初始化，训练时动态创建  

    def train(self, train_loader, val_loader=None, num_epochs=50, **kwargs):  
        current_time = datetime.now().strftime('%H:%M:%S')  
        self.logger.info(f"Start training FCClassifier. Time: {current_time}")  

        scaler = torch.cuda.amp.GradScaler(enabled=self.main_device.startswith('cuda'))  
        scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=0.01,   
            steps_per_epoch=len(train_loader), 
            epochs=num_epochs
        )  

        self.classifier.train()  
        for epoch in range(num_epochs):  
            epoch_start_time = time.time()  
            total_loss = 0.0  

            for images, targets in train_loader:  
                images = images.to(self.main_device)  
                targets = targets.to(self.main_device).float()  

                self.optimizer.zero_grad()  
                
                with torch.cuda.amp.autocast(enabled=self.main_device.startswith('cuda')):  
                    outputs = self.full_model(images)    # 统一走 DataParallel 路径  
                
                loss = self.criterion(outputs, targets)  

                scaler.scale(loss).backward()  
                scaler.step(self.optimizer)  
                scaler.update()  
                scheduler.step()  

                total_loss += loss.item()  

            avg_loss = total_loss / len(train_loader)  
            epoch_duration = time.time() - epoch_start_time  
            self.logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s')  

            if val_loader and (epoch + 1) % 5 == 0:  
                metrics = self.evaluate(val_loader)  
                self.logger.info(f'Epoch {epoch+1} Validation: {metrics}')  
                self.classifier.train()  

        current_time = datetime.now().strftime('%H:%M:%S')  
        self.logger.info(f"FCClassifier training completed. Time: {current_time}")  

    def evaluate(self, data_loader) -> dict:  
        self.classifier.eval()  
        all_labels, all_preds = [], []  

        with torch.no_grad():  
            for images, targets in data_loader:  
                images = images.to(self.main_device)  
                targets_np = targets.cpu().numpy()  
                 
                with torch.cuda.amp.autocast(enabled=self.main_device.startswith('cuda')):  
                    outputs = self.full_model(images)    # 统一走 DataParallel 路径  

                all_labels.append(targets_np)  
                all_preds.append(outputs.sigmoid().cpu().numpy())  

        all_labels = np.concatenate(all_labels, axis=0)  
        all_preds = np.concatenate(all_preds, axis=0)  
        
        f1, f2 = self._calc_metrics(all_labels, all_preds)
        self.logger.info(f'Validation - F1 Score: {f1:.4f}, F2 Score: {f2:.4f}')  

        return {'f1': f1, 'f2': f2}  
    
    def _format_prediction(self, images) -> dict:  
           
        with torch.cuda.amp.autocast(enabled=self.main_device.startswith('cuda')):  
            outputs = self.full_model(images)    # 统一走 DataParallel 路径  
        probs = outputs.sigmoid().cpu().numpy()[0]  
            
        return {  
            'predictions': {f'label_{i}': float(p) for i, p in enumerate(probs)},  
            'top_label': int(np.argmax(probs))  
        }