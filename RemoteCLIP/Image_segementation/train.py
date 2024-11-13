import os  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.cuda.amp import GradScaler, autocast  
import logging  
import numpy as np  
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  
from pathlib import Path  
import time  
from datetime import datetime  

from utils import (
    load_and_save_data, 
    calculate_metrics, 
    CombinedLoss, 
    EarlyStopping
)
from dataset import create_dataloaders  
from model import RemoteClipUNet  
from config import get_config, setup_device, setup_logging  

def train_model(model, train_loader, val_loader, config):  
    """训练模型的主函数"""  
    logging.info("Starting training with configuration:")  
    for key, value in config['training'].items():  
        logging.info(f"{key}: {value}")  

    device = setup_device()  
    model = model.to(device)  
    
    # 优化器  
    optimizer = optim.AdamW(  
        model.parameters(),  
        lr=config['training']['learning_rate'],  
        weight_decay=config['training']['weight_decay']  
    )  
    
    # 学习率调度器  
    scheduler = CosineAnnealingWarmRestarts(  
        optimizer,  
        T_0=config['training']['scheduler_T0'],  
        T_mult=config['training']['scheduler_T_mult'],  
        eta_min=config['training']['min_lr']  
    )  
    
    # 损失函数  
    criterion = CombinedLoss(  
        weights=config['training'].get('loss_weights', [0.5, 0.5]),  
        ignore_index=config['training'].get('ignore_index', 0)  
    )  
    
    # 混合精度训练  
    scaler = GradScaler()  
    
    # 初始化早停  
    early_stopping = EarlyStopping(  
        patience=config['training']['patience'],  
        mode='max'  
    )  
    
    best_miou = float('-inf')  
    
    for epoch in range(config['training']['epochs']):  
        # 训练阶段  
        model.train()  
        epoch_loss = 0  
        epoch_start = time.time()  
        batch_count = 0  
        
        for images, masks in train_loader:  
            images = images.to(device)  
            masks = masks.to(device)  
            
            # 混合精度训练  
            with autocast():  
                outputs = model(images)  
                loss = criterion(outputs, masks)  
            
            optimizer.zero_grad()  
            scaler.scale(loss['total']).backward()  
            
            # 梯度裁剪  
            if config['training']['clip_grad_norm'] > 0:  
                scaler.unscale_(optimizer)  
                torch.nn.utils.clip_grad_norm_(  
                    model.parameters(),  
                    config['training']['clip_grad_norm']  
                )  
            
            scaler.step(optimizer)  
            scaler.update()  
            
            # 累积损失  
            epoch_loss += loss['total'].item()  
            batch_count += 1  
        
        scheduler.step()  
        avg_loss = epoch_loss / batch_count  
        epoch_time = time.time() - epoch_start  
        
        # 验证  
        if (epoch + 1) % config['training']['val_frequency'] == 0:  
            model.eval()  
            val_loss = 0  
            val_metrics = {'accuracy': 0, 'mean_iou': 0}  
            val_batches = 0  
            
            with torch.no_grad():  
                for images, masks in val_loader:  
                    images = images.to(device)  
                    masks = masks.to(device)  
                    
                    outputs = model(images)  
                    loss = criterion(outputs, masks)  
                    val_loss += loss['total'].item()  
                    
                    # 计算指标  
                    batch_metrics = calculate_metrics(  
                        outputs['main'],  
                        masks,  
                        num_classes=config['dataset']['num_classes']  
                    )  
                    
                    val_metrics['accuracy'] += batch_metrics['accuracy']  
                    val_metrics['mean_iou'] += batch_metrics['mean_iou']  
                    val_batches += 1  
            
            # 计算平均指标  
            val_loss /= val_batches  
            val_metrics['accuracy'] /= val_batches  
            val_metrics['mean_iou'] /= val_batches  
            
            # 记录训练和验证结果  
            logging.info(  
                f"Epoch {epoch+1}/{config['training']['epochs']} "  
                f"[{epoch_time:.2f}s] - "  
                f"Train Loss: {avg_loss:.4f}, "  
                f"Val Loss: {val_loss:.4f}, "  
                f"Val Acc: {val_metrics['accuracy']:.4f}, "  
                f"Val mIoU: {val_metrics['mean_iou']:.4f}"  
            )  
            
            # 保存最佳模型  
            if val_metrics['mean_iou'] > best_miou:  
                best_miou = val_metrics['mean_iou']  
                model_path = Path(config['paths']['model']['save_dir']) / 'best_model.pth'  
                torch.save(model.state_dict(), model_path)  
                logging.info(f"Saved best model with mIoU: {best_miou:.4f}")  
            
            # 早停检查  
            if early_stopping(val_metrics['mean_iou']):  
                logging.info(f"Early stopping triggered after {epoch+1} epochs")  
                break  
    
    logging.info("Training completed!")  
    return model  

def main():  
    """主函数"""  
    try:  
        # 配置和日志  
        config = get_config()  
        
        # 创建实验目录  
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  
        exp_dir = Path(config['paths']['model']['save_dir']) / timestamp  
        exp_dir.mkdir(parents=True, exist_ok=True)  
        
        # 设置日志  
        log_path = exp_dir / 'training.log'  
        setup_logging(log_path)  
        
        # 保存配置  
        import json  
        with open(exp_dir / 'config.json', 'w') as f:  
            json.dump(config.config, f, indent=4)  
        
        logging.info("Starting training pipeline...")  
        
        # 加载数据  
        image_path = Path(config['paths']['data']['images']) / config['paths']['input']['train_image']  
        label_path = Path(config['paths']['data']['images']) / config['paths']['input']['train_label']  
        
        image, labels, _ = load_and_save_data(  
            image_path=image_path,  
            label_path=label_path,  
            output_dir=config['paths']['data']['process']  
        )  
        logging.info(f"Data loaded - Image shape: {image.shape}, Labels shape: {labels.shape}")  
        
        # 创建数据加载器  
        train_loader, val_loader = create_dataloaders(  
            image=image,  
            labels=labels,  
            patch_size=config['dataset']['patch_size'],  
            num_patches=config['dataset']['patch_number'],  
            batch_size=config['training']['batch_size'],  
            train_ratio=config['dataset']['train_val_split'],  
            num_workers=config['dataset']['num_workers']  
        )  
        
        # 初始化模型  
        model = RemoteClipUNet(  
            model_name=config['model']['model_name'],  
            ckpt_path=config['paths']['model']['clip_ckpt'],  
            num_classes=config['dataset']['num_classes'],  
            dropout_rate=0.2,  
            initial_features=128
        )  
        
        # 训练模型  
        model = train_model(model, train_loader, val_loader, config)  
        
    except Exception as e:  
        logging.error(f"Error occurred: {str(e)}", exc_info=True)  
        raise  

if __name__ == '__main__':  
    main()