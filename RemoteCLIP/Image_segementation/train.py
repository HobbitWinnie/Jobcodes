import torch  
import torch.optim as optim  
import logging  
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  
from pathlib import Path  
import time  
from datetime import datetime  
import os  
import json  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.cuda.amp import GradScaler, autocast  

from utils import load_and_save_data, calculate_metrics, CombinedLoss, EarlyStopping  
from dataset import create_dataloaders  
from model import RemoteClipUNet  
from config import get_config, setup_logging  

class ModelTrainer:  
    def __init__(self, config):  
        """初始化训练器"""  
        start_time = time.time()  
        print(f"开始设置训练环境... 时间: {datetime.now().strftime('%H:%M:%S')}")  

        self.config = config  

         # 修改设备初始化逻辑  
        if torch.cuda.is_available():  
            self.device = torch.device('cuda')  
            print(f"使用GPU数量: {torch.cuda.device_count()}")  
            for i in range(torch.cuda.device_count()):  
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")  
        else:  
            self.device = torch.device('cpu')  
            print("使用CPU训练")  

        # 训练相关组件  
        self.model = None  
        self.optimizer = None  
        self.scheduler = None  
        self.criterion = None  
        self.scaler = GradScaler()  
        self.best_miou = float('-inf')  

        # 梯度裁剪阈值  
        self.max_grad_norm = config['training'].get('max_grad_norm', 1.0) 
        
        # 确保使用正确的ignore_index  
        self.ignore_index = config['training'].get('ignore_index', 0)  

        self.early_stopping = EarlyStopping(  
            patience=config['training']['patience'],  
            mode='max',
            min_delta=1e-4  # 最小改善阈值  

        )  

        total_init_time = time.time() - start_time  
        print(f"基本初始化完成，耗时: {total_init_time:.2f}秒")  

    def initialize_training(self, image, labels):  
        """初始化训练环境"""  
        start_time = time.time()  
        print(f"开始初始化训练环境... 时间: {datetime.now().strftime('%H:%M:%S')}")  

        # 创建数据加载器  
        print("开始创建数据加载器...")  
        dataloader_start = time.time()  
        
        self.train_loader, self.val_loader = create_dataloaders(  
            image=image,  
            labels=labels,  
            patch_size=self.config['dataset']['patch_size'],  
            num_patches=self.config['dataset']['patch_number'],  
            batch_size=self.config['training']['batch_size'],  
            train_ratio=self.config['dataset']['train_val_split'],  
            num_workers=self.config['dataset']['num_workers']  
        )  
        dataloader_time = time.time() - dataloader_start  
        print(f"数据加载器创建完成，耗时: {dataloader_time:.2f}秒")  

        # 初始化模型  
        print("开始初始化模型...")  
        model_start = time.time()  

        self.model = RemoteClipUNet(  
            model_name=self.config['model']['model_name'],  
            ckpt_path=self.config['paths']['model']['clip_ckpt'],  
            num_classes=self.config['dataset']['num_classes'],  
            dropout_rate=0.2,  
            initial_features=128  
        ).to(self.device)  

        # 检查是否使用多GPU  
        if torch.cuda.device_count() > 1:  
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")  
            self.model = nn.DataParallel(self.model)  
            # 注意：使用DataParallel后，模型的属性会被封装在module中  
            logging.info(f"模型初始化完成，使用 {torch.cuda.device_count()} 个GPU")  
        else:  
            print("使用单GPU或CPU进行训练")  

        # 将模型移动到设备  
        self.model = self.model.to(self.device) 

        model_time = time.time() - model_start  
        print(f"模型初始化完成，耗时: {model_time:.2f}秒")  

        # 初始化优化器和调度器  
        print("开始初始化优化器和调度器...")  
        optimizer_start = time.time()  
        
        self.optimizer = optim.AdamW(  
            self.model.parameters(),  
            lr=self.config['training']['learning_rate'],  
            weight_decay=self.config['training']['weight_decay']  
        )  

        # 使用单一学习率调度器，简化策略  
        self.scheduler = CosineAnnealingWarmRestarts(  
            self.optimizer,  
            T_0=self.config['training']['scheduler_T0'],  
            T_mult=self.config['training']['scheduler_T_mult'],  
            eta_min=self.config['training']['min_lr']  
        )  

        self.criterion = CombinedLoss(  
            weights=self.config['training'].get('loss_weights', [0.5, 0.5]),  
            ignore_index=self.ignore_index,  
        ).to(self.device)  

        optimizer_time = time.time() - optimizer_start  
        print(f"优化器和调度器初始化完成，耗时: {optimizer_time:.2f}秒")  

        total_time = time.time() - start_time  
        print(f"训练环境初始化完成，总耗时: {total_time:.2f}秒")  

    def train_epoch(self):  
        """训练一个epoch"""  
        self.model.train()  
        epoch_loss = 0  
        epoch_start = time.time()  
        batch_count = 0  

        for batch in self.train_loader:  
            # 确保batch是一个包含两个元素的元组  
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:  
                raise ValueError(f"Unexpected batch format. Expected tuple of length 2, got: {type(batch)}")  
            
            try:  
                images, masks = batch  
                images = images.to(self.device, non_blocking=True)  
                masks = masks.to(self.device, non_blocking=True)                       
                
                # 清除梯度  
                self.optimizer.zero_grad(set_to_none=True)  
                
                # 前向传播（使用autocast） 
                with autocast():  
                    outputs = self.model(images)  
                    loss = self.criterion(outputs, masks)  
                
                # 检查损失值  
                if torch.isnan(loss) or torch.isinf(loss):  
                    logging.warning(f"跳过无效的损失值: {loss.item()}")  
                    continue  

                # 反向传播  
                self.scaler.scale(loss).backward()  
                            
                # 检查梯度是否有效  
                valid_gradients = True  
                for param in self.model.parameters():  
                    if param.grad is not None:  
                        if not torch.isfinite(param.grad).all():  
                            valid_gradients = False  
                            break  
                
                if not valid_gradients:  
                    logging.warning("检测到无效梯度，跳过此批次")  
                    # 重置优化器和scaler  
                    self.optimizer.zero_grad(set_to_none=True)  
                    self.scaler = GradScaler()  # 重新创建scaler  
                    continue  
                
                # 梯度裁剪  
                if self.max_grad_norm > 0:  
                    try:  
                        self.scaler.unscale_(self.optimizer)  
                        grad_norm = torch.nn.utils.clip_grad_norm_(  
                            self.model.parameters(),  
                            max_norm=self.max_grad_norm  
                        )  
                        
                        if not torch.isfinite(grad_norm):  
                            logging.warning(f"梯度范数无效: {grad_norm}")  
                            # 重置优化器和scaler  
                            self.optimizer.zero_grad(set_to_none=True)  
                            self.scaler = GradScaler()  # 重新创建scaler  
                            continue  
                            
                    except RuntimeError as e:  
                        if "unscale_() has already been called" in str(e):  
                            logging.warning("检测到重复的unscale调用，重置训练状态")  
                            self.optimizer.zero_grad(set_to_none=True)  
                            self.scaler = GradScaler()  # 重新创建scaler  
                            continue  
                        else:  
                            raise e  
                
                # 更新参数  
                try:  
                    self.scaler.step(self.optimizer)  
                    self.scaler.update()  
                except RuntimeError as e:  
                    logging.warning(f"优化器步骤失败: {str(e)}")  
                    self.optimizer.zero_grad(set_to_none=True)  
                    self.scaler = GradScaler()  # 重新创建scaler  
                    continue  
                
                # 更新损失统计  
                loss_value = loss.item()  
                epoch_loss += loss_value  
                batch_count += 1  
                
            except Exception as e:  
                logging.error(f"批次训练错误: {str(e)}")  
                # 重置训练状态  
                self.optimizer.zero_grad(set_to_none=True)  
                self.scaler = GradScaler()  # 重新创建scaler  
                continue  
        
        if batch_count == 0:  
            logging.warning("本轮训练未完成任何有效批次")  
            return float('inf'), time.time() - epoch_start  
        
        return epoch_loss / batch_count, time.time() - epoch_start  
    
    def validate(self):  
        """验证模型"""  
        self.model.eval()  
        val_loss = 0.0  
        val_metrics = {'accuracy': 0.0, 'mean_iou': 0.0}  
        val_batches = 0  
        
        with torch.no_grad():  
            for batch in self.val_loader:  
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:  
                    raise ValueError(f"Unexpected validation batch format: {type(batch)}")  
                
                images, masks = batch  
                images = images.to(self.device, non_blocking=True)  
                masks = masks.to(self.device, non_blocking=True)  
                
                with autocast():  
                    outputs = self.model(images)  
                    loss = self.criterion(outputs, masks)  
                    val_loss += loss.item()  
                    
                    pred = outputs['main'] if isinstance(outputs, dict) else outputs  
                    batch_metrics = calculate_metrics(  
                        pred,  
                        masks,  
                        num_classes=self.config['dataset']['num_classes']  
                    )  
                    
                    val_metrics['accuracy'] += batch_metrics['accuracy']  
                    val_metrics['mean_iou'] += batch_metrics['mean_iou']  
                    val_batches += 1  

        if val_batches > 0:  
            val_loss /= val_batches  
            val_metrics['accuracy'] /= val_batches  
            val_metrics['mean_iou'] /= val_batches  
            
            # # 打印详细的验证结果  
            # print(f"\nValidation Summary:")  
            # print(f"Average Loss: {val_loss:.4f}")  
            # print(f"Average Accuracy: {val_metrics['accuracy']:.4f}")  
            # print(f"Average mIoU: {val_metrics['mean_iou']:.4f}")  
        else:  
            logging.warning("没有成功处理任何验证批次！")  
        
        return val_loss, val_metrics  

    def save_checkpoint(self, val_metrics, epoch):  
        """保存检查点并检查是否需要早停"""  
        if val_metrics['mean_iou'] > self.best_miou:  
            self.best_miou = val_metrics['mean_iou']  
            model_path = Path(self.config['paths']['model']['save_dir']) / 'best_model.pth'  
            torch.save(self.model.state_dict(), model_path)  
            logging.info(f"保存最佳模型 mIoU: {self.best_miou:.4f}")  

        return self.early_stopping(val_metrics['mean_iou'])  

    def train(self, image, labels):  
        """训练模型的主循环"""  
        try:  
            self.initialize_training(image, labels)  

            print("\n开始训练...")  
            print(f"总轮次: {self.config['training']['epochs']}")  
            print(f"验证频率: 每 {self.config['training']['val_frequency']} 轮")  
            print(f"初始学习率: {self.config['training']['learning_rate']}")  
            print(f"梯度裁剪阈值: {self.max_grad_norm}")  
            print(f"Ignore index: {self.ignore_index}\n")  
            
            for epoch in range(self.config['training']['epochs']):  
                # 获取当前学习率  
                current_lr = self.optimizer.param_groups[0]['lr']  
                
                avg_loss, epoch_time = self.train_epoch()  
                self.scheduler.step()  

                if (epoch + 1) % self.config['training']['val_frequency'] == 0:  
                    val_loss, val_metrics = self.validate() 

                    logging.info(  
                        f"Epoch {epoch+1}/{self.config['training']['epochs']} "  
                        f"[{epoch_time:.2f}s] - "  
                        f"Train Loss: {avg_loss:.4f}, "  
                        f"Val Loss: {val_loss:.4f}, "  
                        f"Val Acc: {val_metrics['accuracy']:.4f}, "  
                        f"Val mIoU: {val_metrics['mean_iou']:.4f}, "  
                        f"LR: {current_lr:.6f}"  
                    )  

                    should_stop = self.save_checkpoint(val_metrics, epoch)  
                    if should_stop:  
                        logging.info(f"Early stopping triggered after {epoch+1} epochs")  
                        break  

                # 检查学习率是否太小  
                if current_lr < 1e-7:  
                    logging.info("Learning rate too small, stopping training")  
                    break  

        except Exception as e:  
            logging.error(f"训练过程出错: {str(e)}")  
            raise  

def main():  
    """主程序入口"""  
    try:  
        # 检查CUDA是否可用  
        print(f"CUDA是否可用: {torch.cuda.is_available()}")  
        if torch.cuda.is_available():  
            print(f"可用GPU数量: {torch.cuda.device_count()}")  
            for i in range(torch.cuda.device_count()):  
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")  

        # 加载配置  
        config = get_config()  
        
        # 创建实验目录  
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  
        exp_dir = Path(config['paths']['model']['save_dir']) / timestamp  
        exp_dir.mkdir(parents=True, exist_ok=True)  

        # 设置日志  
        setup_logging(exp_dir / 'training.log')  
        
        # 保存配置  
        with open(exp_dir / 'config.json', 'w') as f:  
            json.dump(config.config, f, indent=4)  

        # 加载数据  
        print("开始加载数据...")  
        image_path = Path(config['paths']['data']['images']) / config['paths']['input']['train_image']  
        label_path = Path(config['paths']['data']['images']) / config['paths']['input']['train_label']  
        
        print(f"图像路径: {image_path}")  
        print(f"标签路径: {label_path}")  
        
        image, labels, _ = load_and_save_data(  
            image_path=image_path,  
            label_path=label_path,  
            output_dir=config['paths']['data']['process']  
        )  
        print(f"数据加载完成: 图像形状 {image.shape}, 标签形状 {labels.shape}")  

        # 创建训练器并开始训练  
        trainer = ModelTrainer(config)  
        trainer.train(image, labels)  

    except Exception as e:  
        logging.error("训练失败", exc_info=True)  
        raise  

if __name__ == '__main__':  
    main()