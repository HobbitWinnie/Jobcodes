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

from utils import load_and_save_data, calculate_metrics  
from dataset import create_dataloaders  
from model import RemoteClipUNet, create_loss_fn  
from config import get_config, setup_device, setup_logging  

class AverageMeter:  
    """跟踪指标的平均值和当前值"""  
    def __init__(self):  
        self.val = 0  
        self.avg = 0  
        self.sum = 0  
        self.count = 0  

    def reset(self):  
        self.val = 0  
        self.avg = 0  
        self.sum = 0  
        self.count = 0  

    def update(self, val, n=1):  
        self.val = val  
        self.sum += val * n  
        self.count += n  
        self.avg = self.sum / self.count  

@torch.no_grad()  
def validate(model, val_loader, criterion, device, epoch, config):  
    """验证模型"""  
    model.eval()  
    metrics = {  
        'val_loss': AverageMeter(),  
        'pixel_acc': AverageMeter(),  
        'mean_iou': AverageMeter()  
    }  
    
    for images, masks in val_loader:  
        images = images.to(device)  
        masks = masks.to(device)  
        
        outputs = model(images)  
        loss = criterion(outputs, masks)  
        
        batch_metrics = calculate_metrics(  
            outputs['main'],   
            masks,  
            num_classes=config['dataset']['num_classes']  
        )  
        
        metrics['val_loss'].update(loss['total'].item())  
        metrics['pixel_acc'].update(batch_metrics['pixel_acc'])  
        metrics['mean_iou'].update(batch_metrics['mean_iou'])  
    
    return {k: v.avg for k, v in metrics.items()}  

def save_checkpoint(state, save_path, is_best=False):  
    """保存检查点"""  
    save_path = Path(save_path)  
    save_path.parent.mkdir(parents=True, exist_ok=True)  
    
    # 保存最新检查点  
    torch.save(state, save_path)  
    
    # 如果是最佳模型，复制一份  
    if is_best:  
        best_path = save_path.parent / 'model_best.pth'  
        torch.save(state, best_path)  
        logging.info(f"Saved best model with mIoU: {state['best_miou']:.4f}")  

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):  
    """加载检查点"""  
    if not os.path.exists(checkpoint_path):  
        return 0, float('-inf')  
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")  
    checkpoint = torch.load(checkpoint_path)  
    
    model.load_state_dict(checkpoint['model_state_dict'])  
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    if scheduler and 'scheduler_state_dict' in checkpoint:  
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  
    
    return checkpoint['epoch'], checkpoint['best_miou']  

def train_model(model, train_loader, val_loader, config):  
    """训练模型的主函数"""  
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
    criterion = create_loss_fn()  
    
    # 混合精度训练  
    scaler = GradScaler()  
    
    # 加载检查点  
    start_epoch, best_miou = load_checkpoint(  
        model, optimizer, scheduler,  
        os.path.join(config['paths']['model']['save_dir'], 'latest.pth')  
    )  
    
    patience_counter = 0  
    
    for epoch in range(start_epoch, config['training']['epochs']):  
        # 训练阶段  
        model.train()  
        epoch_loss = AverageMeter()  
        epoch_start = time.time()  
        
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
            
            # 更新epoch损失  
            epoch_loss.update(loss['total'].item())  
        
        scheduler.step()  
        
        # 计算epoch训练时间  
        epoch_time = time.time() - epoch_start  
        
        # 验证  
        if (epoch + 1) % config['training']['val_frequency'] == 0:  
            val_metrics = validate(model, val_loader, criterion, device, epoch+1, config)  
            
            # 记录训练和验证结果  
            logging.info(  
                f"Epoch {epoch+1}/{config['training']['epochs']} "  
                f"[{epoch_time:.2f}s] - "  
                f"Train Loss: {epoch_loss.avg:.4f}, "  
                f"Val Loss: {val_metrics['val_loss']:.4f}, "  
                f"Val Acc: {val_metrics['pixel_acc']:.4f}, "  
                f"Val mIoU: {val_metrics['mean_iou']:.4f}"  
            )  
            
            # 保存检查点  
            is_best = val_metrics['mean_iou'] > best_miou  
            if is_best:  
                best_miou = val_metrics['mean_iou']  
                patience_counter = 0  
            else:  
                patience_counter += 1  
            
            save_checkpoint(  
                {  
                    'epoch': epoch + 1,  
                    'model_state_dict': model.state_dict(),  
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'scheduler_state_dict': scheduler.state_dict(),  
                    'best_miou': best_miou,  
                    'config': config  
                },  
                os.path.join(config['paths']['model']['save_dir'], 'latest.pth'),  
                is_best  
            )  
        
        # 早停  
        if patience_counter >= config['training']['patience']:  
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
            json.dump(config, f, indent=4)  
        
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
            remoteclip_path=config['paths']['model']['clip_ckpt'],  
            num_classes=config['dataset']['num_classes']  
        )  
        
        # 训练模型  
        model = train_model(model, train_loader, val_loader, config)  
        
    except Exception as e:  
        logging.error(f"Error occurred: {str(e)}", exc_info=True)  
        raise  

if __name__ == '__main__':  
    main()