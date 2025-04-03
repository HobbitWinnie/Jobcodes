import sys  
sys.path.append('/home/nw/Codes')  

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  

from Models.Pixel_based_CNN_classification.ResNet50 import ResNet50
from dataset import prepare_dataset, RemoteSensingDataset  
from config import ModelConfig  

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = self._prepare_model(model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        self.device = torch.device(config.device)

    def _prepare_model(self, model) -> nn.Module:
        """多GPU并行处理"""
        if torch.cuda.device_count() > 1:
            logger.info(f"使用 {torch.cuda.device_count()} 块GPU")
            return nn.DataParallel(model)
        return model.to(self.device)

    def _create_dataloader(self, dataset, shuffle=True) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size * torch.cuda.device_count(),
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """单epoch训练"""
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """模型验证"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        return accuracy_score(all_labels, all_preds)

    def save_model(self):
        """保存最终模型"""
        save_path = self.config.model_path
        torch.save(
            self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            save_path
        )
        logger.info(f"模型已保存至 {save_path}")

    def run_training(self, train_dataset, val_dataset):
        """执行完整训练流程"""
        train_loader = self._create_dataloader(train_dataset)
        val_loader = self._create_dataloader(val_dataset, shuffle=False)

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc = self.validate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}%"
            )
        
        self.save_model()


def main():  
    # 初始化配置  
    config = ModelConfig()  

    # 准备数据集  
    X, y, nodata_value = prepare_dataset(  
        image_path=config.train_image_path,  
        label_path=config.label_image_path,  
        save_dir=config.sample_dir,  
        sample_size=config.sample_size,  
        patch_size=config.patch_size  
    ) 

    # 划分训练验证集  
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.test_size, 
        random_state=42
    )  

    # 创建数据集  
    train_dataset = RemoteSensingDataset(X_train, y_train)  
    val_dataset = RemoteSensingDataset(X_val, y_val)  

    # 初始化模型  
    model = ResNet50(num_classes=config.num_classes)

    # 启动训练  
    trainer = ModelTrainer(config, model)  
    trainer.run_training(train_dataset, val_dataset)  


if __name__ == "__main__":  
    main()  