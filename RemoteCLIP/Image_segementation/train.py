import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import logging
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
import time
from datetime import datetime
import os
import json
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    load_and_save_data, 
    calculate_metrics, 
    # CombinedLoss, 
    EarlyStopping
)
from dataset import create_dataloaders
from model import RemoteClipUNet
from config import get_config, setup_logging

class CombinedLoss(nn.Module):  
    def __init__(self, weights=[0.5, 0.5], ignore_index=0):  
        super().__init__()  
        self.weights = weights  
        self.ignore_index = ignore_index  
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)  

    def dice_loss(self, pred, target):  
        # 计算dice loss  
        pred = F.softmax(pred, dim=1)  
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  
        
        # 忽略特定类别  
        mask = (target != self.ignore_index).float()  
        pred = pred * mask.unsqueeze(1)  
        target_one_hot = target_one_hot.float() * mask.unsqueeze(1)  
        
        # 计算dice系数  
        intersection = (pred * target_one_hot).sum(dim=(2, 3))  
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) + 1e-7  
        
        return 1 - (2. * intersection / union).mean()  

    def forward(self, outputs, target):  
        if isinstance(outputs, dict):  
            pred = outputs['main']  
        else:  
            pred = outputs  
            
        ce_loss = self.ce(pred, target)  
        dice = self.dice_loss(pred, target)  
        
        return self.weights[0] * ce_loss + self.weights[1] * dice

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'  
    os.environ['MASTER_PORT'] = '12355'  
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

# ... (前面的import部分保持不变)  

def init_training(rank, world_size, config, image, labels):  
    """初始化训练环境和模型"""  
    setup(rank, world_size)  
    torch.cuda.set_device(rank)  

    # 创建数据加载器  
    train_loader, val_loader = create_dataloaders(  
        image=image,  
        labels=labels,  
        patch_size=config['dataset']['patch_size'],  
        num_patches=config['dataset']['patch_number'],  
        batch_size=config['training']['batch_size'],  
        train_ratio=config['dataset']['train_val_split'],  
        num_workers=config['dataset']['num_workers'],  
        rank=rank,  
        world_size=world_size  
    )  

    # 创建和初始化模型  
    model = RemoteClipUNet(  
        model_name=config['model']['model_name'],  
        ckpt_path=config['paths']['model']['clip_ckpt'],  
        num_classes=config['dataset']['num_classes'],  
        dropout_rate=0.2,  
        initial_features=128  
    ).to(rank)  
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)  

    # 初始化训练组件  
    optimizer = optim.AdamW(  
        model.parameters(),  
        lr=config['training']['learning_rate'],  
        weight_decay=config['training']['weight_decay']  
    )  

    scheduler = CosineAnnealingWarmRestarts(  
        optimizer,  
        T_0=config['training']['scheduler_T0'],  
        T_mult=config['training']['scheduler_T_mult'],  
        eta_min=config['training']['min_lr']  
    )  

    criterion = CombinedLoss(  
        weights=config['training'].get('loss_weights', [0.5, 0.5]),  
        ignore_index=config['training'].get('ignore_index', 0),  
    ).to(rank)  

    return model, optimizer, scheduler, criterion, train_loader, val_loader  

def train_epoch(model, train_loader, optimizer, criterion, scaler, rank, world_size, config):  
    """执行一个训练epoch"""  
    model.train()  
    epoch_loss = 0  
    epoch_start = time.time()  
    batch_count = 0  

    for images, masks in train_loader:  
        images = images.to(rank, non_blocking=True)  
        masks = masks.to(rank, non_blocking=True)  

        with autocast():  
            outputs = model(images)  
            loss = criterion(outputs, masks)  

        optimizer.zero_grad(set_to_none=True)  
        scaler.scale(loss).backward()  
        scaler.step(optimizer)  
        scaler.update()  

        # 同步损失  
        epoch_loss += loss.item()   
        batch_count += 1  

    avg_loss = epoch_loss / batch_count  
    epoch_time = time.time() - epoch_start  
    
    return avg_loss, epoch_time  

def validate(model, val_loader, criterion, rank, world_size, config):  
    """执行验证"""  
    model.eval()  
    val_loss = 0  
    val_metrics = {'accuracy': 0, 'mean_iou': 0}  
    val_batches = 0  

    with torch.no_grad():  
        for images, masks in val_loader:  
            images = images.to(rank, non_blocking=True)  
            masks = masks.to(rank, non_blocking=True)  

            outputs = model(images)  
            loss = criterion(outputs, masks)  
            val_loss += loss['total'].item()  

            # 计算指标时使用主输出  
            if isinstance(outputs, dict):  
                pred = outputs['main']  
            else:  
                pred = outputs  
                
            batch_metrics = calculate_metrics(  
                pred,  
                masks,  
                num_classes=config['dataset']['num_classes']  
            )  

            val_metrics['accuracy'] += batch_metrics['accuracy']  
            val_metrics['mean_iou'] += batch_metrics['mean_iou']  
            val_batches += 1  

    # 计算平均值  
    val_loss /= val_batches  
    val_metrics['accuracy'] /= val_batches  
    val_metrics['mean_iou'] /= val_batches  

    return val_loss, val_metrics  

def save_and_log(rank, epoch, config, model, val_metrics, best_miou,   
                 avg_loss, val_loss, epoch_time, early_stopping):  
    """保存模型和记录日志"""  
    if rank == 0:  
        logging.info(  
            f"Epoch {epoch+1}/{config['training']['epochs']} "  
            f"[{epoch_time:.2f}s] - "  
            f"Train Loss: {avg_loss:.4f}, "  
            f"Val Loss: {val_loss:.4f}, "  
            f"Val Acc: {val_metrics['accuracy']:.4f}, "  
            f"Val mIoU: {val_metrics['mean_iou']:.4f}"  
        )  

        if val_metrics['mean_iou'] > best_miou:  
            best_miou = val_metrics['mean_iou']  
            model_path = Path(config['paths']['model']['save_dir']) / 'best_model.pth'  
            torch.save(model.module.state_dict(), model_path)  
            logging.info(f"Saved best model with mIoU: {best_miou:.4f}")  

        should_stop = early_stopping(val_metrics['mean_iou'])  
    else:  
        should_stop = False  
        
    # 广播早停信号  
    should_stop = torch.tensor(1 if should_stop else 0, device=rank)  
    dist.broadcast(should_stop, src=0)  
    
    return should_stop.item(), best_miou  

def train_model_ddp(rank, world_size, config, image, labels):  
    """分布式训练模型的主函数"""  
    try:  
        # 初始化训练环境和组件  
        model, optimizer, scheduler, criterion, train_loader, val_loader = init_training(  
            rank, world_size, config, image, labels  
        )  
        
        # 初始化训练工具  
        scaler = GradScaler()  
        if rank == 0:  
            early_stopping = EarlyStopping(  
                patience=config['training']['patience'],  
                mode='max'  
            )  
        best_miou = float('-inf')  

        # 训练循环  
        for epoch in range(config['training']['epochs']):  
            # 设置采样器epoch  
            train_loader.sampler.set_epoch(epoch)  
            val_loader.sampler.set_epoch(epoch)  

            # 训练一个epoch  
            avg_loss, epoch_time = train_epoch(  
                model, train_loader, optimizer, criterion,   
                scaler, rank, world_size, config  
            )  
            
            scheduler.step()  

            # 验证  
            if (epoch + 1) % config['training']['val_frequency'] == 0:  
                val_loss, val_metrics = validate(  
                    model, val_loader, criterion,   
                    rank, world_size, config  
                )  

                # 保存模型和记录日志  
                should_stop, best_miou = save_and_log(  
                    rank, epoch, config, model, val_metrics,  
                    best_miou, avg_loss, val_loss, epoch_time,  
                    early_stopping if rank == 0 else None  
                )  

                if should_stop:  
                    if rank == 0:  
                        logging.info(f"Early stopping triggered after {epoch+1} epochs")  
                    break  

            # 确保所有进程同步  
            dist.barrier()  

    except Exception as e:  
        logging.error(f"Error in rank {rank}: {str(e)}", exc_info=True)  
        raise  
    finally:  
        cleanup()  

def main():  
    """主函数"""  
    try:  
        # 配置和日志  
        config = get_config()  
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'  

        # 创建实验目录  
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  
        exp_dir = Path(config['paths']['model']['save_dir']) / timestamp  
        exp_dir.mkdir(parents=True, exist_ok=True)  

        # 设置日志  
        log_path = exp_dir / 'training.log'  
        setup_logging(log_path)  

        # 保存配置  
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

        # 获取GPU数量  
        world_size = torch.cuda.device_count()  
        logging.info(f"Using {world_size} GPUs")  

        # 启动多进程训练  
        mp.spawn(  
            train_model_ddp,  
            args=(world_size, config, image, labels),  
            nprocs=world_size,  
            join=True  
        )  

    except Exception as e:  
        logging.error("Training failed", exc_info=True)  
        raise  

if __name__ == '__main__':  
    main()