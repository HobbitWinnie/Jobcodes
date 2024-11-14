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
from config import get_config, setup_device, setup_logging

class CombinedLoss(nn.Module):  
    """增强的组合损失函数：处理所有输出并确保梯度传播"""  

    def __init__(self, weights=[0.5, 0.5], ignore_index=0, aux_weight=0.4):  
        super().__init__()  
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)  
        self.weights = weights  
        self.ignore_index = ignore_index  
        self.aux_weight = aux_weight  
        
        # 添加调试日志  
        logging.info(f"Initializing CombinedLoss with weights={weights}, "  
                    f"ignore_index={ignore_index}, aux_weight={aux_weight}")  

    def compute_single_output_loss(self, pred, target, name=""):  
        """计算单个输出的损失"""  
        # 确保输入是有效的张量  
        assert torch.is_tensor(pred), f"Prediction {name} must be a tensor"  
        assert torch.is_tensor(target), f"Target must be a tensor"  
        
        # 记录shapes用于调试  
        if name:  
            logging.debug(f"{name} shapes - pred: {pred.shape}, target: {target.shape}")  

        # 计算交叉熵损失  
        ce_loss = self.ce(pred, target)  
        
        # 计算Dice损失  
        pred_soft = F.softmax(pred, dim=1)  
        dice_loss = 1 - dice_coefficient(pred_soft, target, self.ignore_index)  
        
        # 确保dice_loss是标量张量  
        if not torch.is_tensor(dice_loss):  
            dice_loss = torch.tensor(dice_loss, device=pred.device, dtype=pred.dtype)  
        
        # 组合损失  
        combined_loss = self.weights[0] * ce_loss + self.weights[1] * dice_loss  
        
        # 返回损失和详细指标  
        metrics = {  
            f"{name}_ce_loss": ce_loss.item(),  
            f"{name}_dice_loss": dice_loss.item(),  
            f"{name}_combined_loss": combined_loss.item()  
        }  
        
        return combined_loss, metrics  

    def forward(self, outputs, target):  
        """前向传播计算损失"""  
        losses = {}  
        metrics = {}  
        total_loss = 0.0  

        # 确保输入是字典  
        if not isinstance(outputs, dict):  
            raise ValueError("Expected outputs to be a dictionary")  

        # 处理主输出的损失  
        if 'main' not in outputs:  
            raise ValueError("Main output not found in model outputs")  
            
        main_loss, main_metrics = self.compute_single_output_loss(  
            outputs['main'], target, "main"  
        )  
        losses['main'] = main_loss  
        metrics.update(main_metrics)  
        total_loss = main_loss  

        # 处理辅助输出的损失  
        if 'aux' in outputs:  
            aux_loss, aux_metrics = self.compute_single_output_loss(  
                outputs['aux'], target, "aux"  
            )  
            losses['aux'] = aux_loss  
            metrics.update(aux_metrics)  
            total_loss = total_loss + self.aux_weight * aux_loss  

        # 处理其他输出，确保梯度流动  
        for key, value in outputs.items():  
            if key not in ['main', 'aux']:  
                if torch.is_tensor(value) and value.requires_grad:  
                    # 添加一个小的损失确保梯度流动  
                    dummy_loss = value.mean() * 0.0  
                    losses[f'dummy_{key}'] = dummy_loss  
                    total_loss = total_loss + dummy_loss  

        # 记录总损失  
        losses['total'] = total_loss  
        metrics['total_loss'] = total_loss.item()  

        # 输出详细的调试信息  
        logging.debug(f"Loss metrics: {metrics}")  
        
        # 验证所有损失都是有效的张量  
        for name, loss in losses.items():  
            if not torch.is_tensor(loss):  
                raise ValueError(f"Loss {name} is not a tensor: {type(loss)}")  
            if torch.isnan(loss):  
                raise ValueError(f"Loss {name} is NaN")  
            if torch.isinf(loss):  
                raise ValueError(f"Loss {name} is Inf")  

        return losses
def dice_coefficient(pred: torch.Tensor,   
                    target: torch.Tensor,   
                    ignore_index: int = 0,   
                    epsilon: float = 1e-6) -> torch.Tensor:  
    """  
    计算Dice系数  
    Args:  
        pred: (B, C, H, W) softmax后的预测  
        target: (B, H, W) 目标  
        ignore_index: 忽略的类别索引  
        epsilon: 防止除零的小值  
    """  
    # 确保输入维度正确  
    assert pred.dim() == 4, f"Prediction must be 4D (B,C,H,W), got {pred.dim()}D"  
    assert target.dim() == 3, f"Target must be 3D (B,H,W), got {target.dim()}D"  
    
    # 获取维度信息  
    b, c, h, w = pred.size()  
    
    # 转换target为one-hot编码  
    target_one_hot = F.one_hot(target, num_classes=c).permute(0, 3, 1, 2).float()  
    
    # 创建mask忽略特定类别  
    mask = (target != ignore_index).float().unsqueeze(1).expand_as(pred)  
    
    # 应用mask  
    pred = pred * mask  
    target_one_hot = target_one_hot * mask  
    
    # 计算交集和并集  
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  
    
    # 计算每个类别的dice系数  
    dice = (2. * intersection + epsilon) / (union + epsilon)  
    
    # 返回所有类别的平均dice系数（忽略背景类）  
    return dice[:, 1:].mean()  # 忽略背景类（假设在索引0）  

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
        aux_weight=0.4  
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
        scaler.scale(loss['total']).backward()  

        if config['training']['clip_grad_norm'] > 0:  
            scaler.unscale_(optimizer)  
            torch.nn.utils.clip_grad_norm_(  
                model.parameters(),  
                config['training']['clip_grad_norm']  
            )  

        scaler.step(optimizer)  
        scaler.update()  

        # 同步损失  
        loss_value = loss['total'].item()  
        dist.all_reduce(torch.tensor(loss_value).to(rank))  
        epoch_loss += loss_value / world_size  
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

            batch_metrics = calculate_metrics(  
                outputs['main'],  
                masks,  
                num_classes=config['dataset']['num_classes']  
            )  

            val_metrics['accuracy'] += batch_metrics['accuracy']  
            val_metrics['mean_iou'] += batch_metrics['mean_iou']  
            val_batches += 1  

    # 同步验证结果  
    val_loss = torch.tensor(val_loss).to(rank)  
    dist.all_reduce(val_loss)  
    val_loss = (val_loss.item() / val_batches) / world_size  

    for key in val_metrics:  
        metric_tensor = torch.tensor(val_metrics[key]).to(rank)  
        dist.all_reduce(metric_tensor)  
        val_metrics[key] = (metric_tensor.item() / val_batches) / world_size  

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