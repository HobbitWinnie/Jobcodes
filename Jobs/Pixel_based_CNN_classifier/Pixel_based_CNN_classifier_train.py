import sys  
sys.path.append('/home/nw/Codes')  

import logging
import torch
import os
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  
from sklearn.metrics import accuracy_score
from Models.CNN_Pixel_based_Classification.ResNet50 import ResNet50
from dataset_util import prepare_dataset, get_dataloaders


logger = logging.getLogger(__name__)


class TrainConfig:
    """模型训练配置参数"""
    
    def __init__(
        self,
        data_root: Path = Path("/home/Dataset/nw/Segmentation/CpeosTest"),
        model_save_path: Path = Path("/home/nw/Codes/Jobs/Pixel_based_CNN_classifier/model_save"),
        num_classes: int = 10,
        batch_size: int = 192,
        num_epochs: int = 500,
        learning_rate: float = 0.001,
        test_size: float = 0.5,
        sample_size: int = 50000,
        patch_size: int = 11,
        weight_decay: float = 1e-4,  
        warmup_epochs: int = 5,  
        gpu_ids = [2, 3],  # 指定使用GPU  
    ):
        # 自动创建保存目录  
        model_save_path.mkdir(parents=True, exist_ok=True)  
        
        # 设备配置  
        self.gpu_ids = self._validate_gpu_ids(gpu_ids)  # GPU ID验证  

        # 路径配置
        self.train_image_path = data_root / "images/GF2_train_image.tif"
        self.label_image_path = data_root / "images/train_label.tif"
        self.sample_dir = data_root / "samples"
        self.model_path = model_save_path / "model_ResNet50_500epoch.pth"

        # 训练参数
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.gpu_ids = gpu_ids  
        self.weight_decay = weight_decay  
        self.warmup_epochs = warmup_epochs  
        
        # 数据采样参数
        self.sample_size = sample_size
        self.patch_size = patch_size

    def _validate_gpu_ids(self, gpu_ids):  
        """验证GPU ID有效性"""  
        if gpu_ids is None:  
            return None  
            
        if not isinstance(gpu_ids, list):  
            raise ValueError("gpu_ids必须为列表类型，例如[0,1]")  
            
        available_gpus = list(range(torch.cuda.device_count()))  
        valid_ids = [i for i in gpu_ids if i in available_gpus]  
        
        if not valid_ids:  
            raise ValueError(f"没有可用的指定GPU，可用设备ID: {available_gpus}")  
            
        logger.info(f"有效GPU设备: {valid_ids}")  
        return valid_ids  


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config, model):
        self.config = config
        self.device = self._select_device()  # 设备选择  
        self.model = self._prepare_model(model)
        self.scaler = GradScaler()  
        
        # 优化器配置  
        self.optimizer = optim.AdamW(  
            self.model.parameters(),  
            lr=config.learning_rate,  
            weight_decay=config.weight_decay  
        )  
        
        # 学习率调度  
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(  
            self.optimizer,   
            mode='max',   
            factor=0.5,  
            patience=5,  
            verbose=True  
        )  
        
        self.criterion = nn.CrossEntropyLoss()  

    def _select_device(self):  
        """设备选择逻辑"""  
        if self.config.gpu_ids:  
            # 手动指定模式  
            main_device = f"cuda:{self.config.gpu_ids[0]}"  
            logger.info(f"手动选择GPU设备: {self.config.gpu_ids}")  
            return torch.device(main_device)  
            
        # 自动检测模式  
        if torch.cuda.is_available():  
            auto_ids = list(range(torch.cuda.device_count()))  
            logger.info(f"自动检测到GPU设备: {auto_ids}")  
            return torch.device(f"cuda:{auto_ids[0]}")  
            
        logger.warning("未检测到可用GPU，使用CPU训练")  
        return torch.device("cpu")  
    
    def _prepare_model(self, model):  
        """模型设备配置"""  
        model = model.to(self.device)  
        
        # 多GPU并行处理  
        if self.config.gpu_ids and len(self.config.gpu_ids) > 1:  
            logger.info(f"启用DataParallel，使用GPU: {self.config.gpu_ids}")  
            return nn.DataParallel(model, device_ids=self.config.gpu_ids)  
            
        return model  
    
    def _warmup_lr(self, epoch):  
        """学习率预热"""  
        if epoch < self.config.warmup_epochs:  
            lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs  
            for param_group in self.optimizer.param_groups:  
                param_group['lr'] = lr  

    def train_epoch(self, train_loader: DataLoader) -> float:
        """单epoch训练"""
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)  
            labels = labels.to(self.device, non_blocking=True)  
            
            self.optimizer.zero_grad()

            with autocast():  
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()  
            self.scaler.step(self.optimizer)  
            self.scaler.update()  
            
            running_loss += loss.item()
            
        return running_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """模型验证"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)  
                labels = labels.to(self.device)  
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)  
        self.scheduler.step(acc)  
        return acc  

    def save_model(self):
        """保存最终模型"""
        save_path = self.config.model_path
        torch.save(
            self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            save_path
        )
        logger.info(f"模型已保存至 {save_path}")

    def run_training(self, train_loader, val_loader):
        """主训练循环"""  
        best_acc = 0.0  
        
        try:  
            for epoch in range(self.config.num_epochs):  
                self._warmup_lr(epoch)  
                
                train_loss = self.train_epoch(train_loader)  
                val_acc = self.validate(val_loader)  
                
                if val_acc > best_acc:  
                    best_acc = val_acc  
                    self.save_model()  
                
                logger.info(  
                    f"Epoch {epoch+1}/{self.config.num_epochs} | "  
                    f"Loss: {train_loss:.4f} | "  
                    f"Acc: {val_acc*100:.2f}% | "  
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"  
                )  
                
        except KeyboardInterrupt:  
            logger.info("训练被用户中断")  
        except Exception as e:  
            logger.error(f"训练异常: {str(e)}")  
            raise  
            
        logger.info(f"训练完成，最佳准确率: {best_acc*100:.2f}%")  


def main():  
    # 初始化配置  
    model_save_path= Path("/home/nw/Codes/Jobs/Pixel_based_CNN_classifier/model_save")
    config = TrainConfig(
        gpu_ids=[2, 3],   # 指定GPU  
        batch_size=256,  
        num_epochs=100,
        model_save_path = model_save_path
    )  

    # 配置日志系统  
    logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
    handlers=[  
        logging.FileHandler(model_save_path / "training.log"),  
        logging.StreamHandler()  
    ]  
)  
    try:  
        # 数据准备  

        X, y, _ = prepare_dataset(  
            image_path=config.train_image_path,  
            label_path=config.label_image_path,  
            save_dir=config.sample_dir,  
            sample_size=config.sample_size,  
            patch_size=config.patch_size,  
        )  

        print("标签值范围:", y.min(), y.max())  
        print("输入数据范围:", X.min(), X.max(), X.mean(), X.std())  

        # 数据加载器  
        train_loader, val_loader = get_dataloaders(  
            patches=X,  
            labels=y,  
            batch_size=config.batch_size,  
            test_size=config.test_size,  
            num_workers=8,  
            pin_memory=True  
        )  

        # 初始化模型  
        model = ResNet50(num_classes=config.num_classes)  

        # 启动训练  
        trainer = ModelTrainer(config, model)  
        trainer.run_training(train_loader, val_loader)  

    except FileNotFoundError as e:  
        logger.error(f"数据文件异常: {str(e)}")  
    except RuntimeError as e:  
        if "CUDA out of memory" in str(e):  
            logger.error("显存不足，建议：\n1. 减小batch_size\n2. 使用更少GPU\n3. 减小输入尺寸")  
        else:  
            logger.error(f"运行时错误: {str(e)}")  

if __name__ == "__main__":  
    main()  