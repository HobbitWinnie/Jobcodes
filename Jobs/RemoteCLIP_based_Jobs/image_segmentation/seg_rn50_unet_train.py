import os  
import sys  
sys.path.append('/home/nw/Codes')  

import torch
import torch.optim as optim
import logging
import time
import numpy as np
import json
import torch.nn as nn
import gc  
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from nw.Codes.Models.RemoteCLIP_based_Segmentation.modules.combined_loss import CombinedLoss

from data.dataset import create_dataloaders
from config import get_config
from utils.set_logging import setup_logging
from Models.RemoteCLIP_based_Segmentation.factory import segmentation_model_factory

def init_training(config):  
    """初始化训练组件"""  
    # 设置设备并确保cudnn基准测试  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    if torch.cuda.is_available():  
        torch.backends.cudnn.benchmark = True  
        torch.backends.cudnn.deterministic = False  
        logging.info(f"CUDA版本: {torch.version.cuda}")  
        logging.info(f"可用GPU: {torch.cuda.get_device_name(0)}")  
        logging.info(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")  

    # 创建实验目录  
    exp_dir = Path(config['paths']['model']['save_dir'])
    exp_dir.mkdir(parents=True, exist_ok=True)  

    # 设置日志  
    setup_logging(exp_dir)  
    with open(exp_dir / 'config.json', 'w') as f:  
        json.dump(config.config, f, indent=4)  

    # 初始化优化器  
    optimizer = optim.AdamW(  
        model.parameters(),  
        lr=config['training']['learning_rate'],  
        weight_decay=config['training']['weight_decay'],  
        betas=(0.9, 0.999)  
    )  

    # 初始化学习率调度器  
    scheduler = CosineAnnealingWarmRestarts(  
        optimizer,  
        T_0=config['training']['scheduler_T0'],  
        T_mult=config['training']['scheduler_T_mult'],  
        eta_min=config['training']['min_lr']  
    )  

    # 初始化损失函数  
    criterion = CombinedLoss(  
        gamma=config['training'].get('gamma', 2.0),  
        alpha=config['training'].get('alpha', 0.5),  
        ignore_index=config['training']['ignore_index']  
    ).to(device)  

    return device, exp_dir, optimizer, scheduler, criterion  

def validate_model(model, val_loader, criterion, device, num_classes):  
    """验证模型性能"""  
    model.eval()  
    val_loss = 0  
    loss_components = {'focal_loss': 0, 'dice_loss': 0}  
    class_metrics = {i: {'correct': 0, 'total': 0} for i in range(num_classes)}  
    confusion_matrix = np.zeros((num_classes, num_classes))  

    with torch.no_grad():  
        for batch in val_loader:  
            images, masks = batch[0].to(device), batch[1].to(device)  

            with autocast():  
                outputs = model(images)  
                loss, loss_info = criterion(outputs, masks)  

            val_loss += loss.item()  
            loss_components['focal_loss'] += loss_info['focal_loss']  
            loss_components['dice_loss'] += loss_info['dice_loss']  

            preds = outputs['main'].argmax(1) if isinstance(outputs, dict) else outputs.argmax(1)  

            # 更新混淆矩阵  
            for true, pred in zip(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten()):  
                confusion_matrix[true][pred] += 1  

            # 更新类别指标  
            for cls in range(num_classes):  
                mask = masks == cls  
                class_metrics[cls]['correct'] += ((preds == cls) & mask).sum().item()  
                class_metrics[cls]['total'] += mask.sum().item()  

            # 清理GPU内存  
            del outputs, loss  
            torch.cuda.empty_cache()  

    # 计算平均损失  
    num_batches = len(val_loader)  
    avg_loss = val_loss / num_batches  
    avg_focal_loss = loss_components['focal_loss'] / num_batches  
    avg_dice_loss = loss_components['dice_loss'] / num_batches  

    # 计算每个类别的性能指标  
    class_performance = {}  
    for cls in range(num_classes):  
        metrics = class_metrics[cls]  
        if metrics['total'] > 0:  
            accuracy = metrics['correct'] / metrics['total']  
            true_positive = confusion_matrix[cls][cls]  
            false_positive = confusion_matrix[:, cls].sum() - true_positive  
            false_negative = confusion_matrix[cls, :].sum() - true_positive  

            iou = true_positive / (true_positive + false_positive + false_negative + 1e-10)  
            precision = true_positive / (true_positive + false_positive + 1e-10)  
            recall = true_positive / (true_positive + false_negative + 1e-10)  
            f1 = 2 * precision * recall / (precision + recall + 1e-10)  

            class_performance[cls] = {  
                'accuracy': float(accuracy),  
                'iou': float(iou),  
                'precision': float(precision),  
                'recall': float(recall),  
                'f1': float(f1),  
                'sample_count': int(metrics['total'])  
            }  

    # 计算总体指标  
    total_correct = sum(m['correct'] for m in class_metrics.values())  
    total_samples = sum(m['total'] for m in class_metrics.values())  
    accuracy = total_correct / total_samples if total_samples > 0 else 0  
    mean_iou = np.mean([p['iou'] for p in class_performance.values()])  

    validation_results = {  
        'loss': avg_loss,  
        'focal_loss': avg_focal_loss,  
        'dice_loss': avg_dice_loss,  
        'accuracy': accuracy,  
        'mean_iou': mean_iou,  
        'class_performance': class_performance,  
        'confusion_matrix': confusion_matrix.tolist()  
    }  

    return validation_results  

def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config, exp_dir):  
    """训练循环"""  
    scaler = GradScaler()  
    best_miou = float('-inf')  
    total_epochs = config['training']['epochs']  
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)  

    # 记录训练历史  
    metrics_history = {  
        'train_loss': [],  
        'train_focal_loss': [],  
        'train_dice_loss': [],  
        'val_loss': [],  
        'val_focal_loss': [],  
        'val_dice_loss': [],  
        'val_miou': [],  
        'val_accuracy': [],  
        'learning_rate': []  
    }  

    logging.info(f"开始训练 - 总轮次: {total_epochs}")  

    try:  
        for epoch in range(total_epochs):  
            model.train()  
            epoch_loss = 0  
            epoch_focal_loss = 0  
            epoch_dice_loss = 0  
            batch_count = 0  
            epoch_start = time.time()  
            current_lr = optimizer.param_groups[0]['lr']  

            for batch in train_loader:  
                images, masks = batch[0].to(device), batch[1].to(device)  

                optimizer.zero_grad(set_to_none=True)  

                with autocast():  
                    outputs = model(images)  
                    loss, loss_info = criterion(outputs, masks)  

                if not torch.isfinite(loss):  
                    logging.warning(f"检测到非有限损失值: {loss.item()}")  
                    continue  

                scaler.scale(loss).backward()  

                for name, param in model.named_parameters():  
                    if param.grad is not None:  
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():  
                            logging.warning(f"梯度中检测到 NaN 或 Inf，参数: {name}")  
                    if torch.isnan(param).any() or torch.isinf(param).any():  
                        logging.warning(f"参数中检测到 NaN 或 Inf，参数: {name}")

                # 添加梯度剪切  
                if max_grad_norm > 0:  
                    scaler.unscale_(optimizer)  
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  

                scaler.step(optimizer)  
                scaler.update()  

                # 更新损失统计  
                epoch_loss += loss.item()  
                epoch_focal_loss += loss_info['focal_loss']  
                epoch_dice_loss += loss_info['dice_loss']  
                batch_count += 1  

                # 清理内存  
                del outputs, loss  
                torch.cuda.empty_cache()  

            # 计算平均损失  
            avg_loss = epoch_loss / batch_count  
            avg_focal_loss = epoch_focal_loss / batch_count  
            avg_dice_loss = epoch_dice_loss / batch_count  
            epoch_time = time.time() - epoch_start  

            # 更新学习率  
            scheduler.step()  

            # 记录训练指标  
            metrics_history['train_loss'].append(avg_loss)  
            metrics_history['train_focal_loss'].append(avg_focal_loss)  
            metrics_history['train_dice_loss'].append(avg_dice_loss)  
            metrics_history['learning_rate'].append(current_lr)  

            # 验证  
            val_metrics = validate_model(  
                model, val_loader, criterion, device,  
                config['dataset']['num_classes']  
            )  

            # 更新验证指标历史  
            metrics_history['val_loss'].append(val_metrics['loss'])  
            metrics_history['val_focal_loss'].append(val_metrics['focal_loss'])  
            metrics_history['val_dice_loss'].append(val_metrics['dice_loss'])  
            metrics_history['val_miou'].append(val_metrics['mean_iou'])  
            metrics_history['val_accuracy'].append(val_metrics['accuracy'])  

            # 输出详细的训练信息  
            logging.info(  
                f"Epoch {epoch + 1}/{total_epochs} [{epoch_time:.2f}s], "  
                f"Training: [{avg_loss:.4f} (Focal: {avg_focal_loss:.4f}, Dice: {avg_dice_loss:.4f})], "  
                f"Validation: [{val_metrics['loss']:.4f} (Focal: {val_metrics['focal_loss']:.4f}, Dice: {val_metrics['dice_loss']:.4f}), "  
                f"Acc = {val_metrics['accuracy']:.4f}, mIoU = {val_metrics['mean_iou']:.4f}], LR: {current_lr:.6f}"  
            )  

            # 保存检查点  
            checkpoint = {  
                'epoch': epoch + 1,  
                'model_state_dict': model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),  
                'scheduler_state_dict': scheduler.state_dict(),  
                'scaler_state_dict': scaler.state_dict(),  
                'best_miou': best_miou,  
                'metrics_history': metrics_history,  
                'config': config.config  
            }  

            # 保存最佳模型  
            if val_metrics['mean_iou'] > best_miou:  
                best_miou = val_metrics['mean_iou']  
                torch.save(checkpoint, exp_dir / 'best_model.pth')  
                logging.info(f"保存最佳模型 (mIoU: {best_miou:.4f})")  

            # 保存训练指标  
            with open(exp_dir / 'metrics_history.json', 'w') as f:  
                json.dump(metrics_history, f, indent=4)  

            # 学习率检查  
            if current_lr < 1e-7:  
                logging.info("学习率过小，停止训练")  
                break  

            # 垃圾回收  
            gc.collect()  

    except KeyboardInterrupt:  
        logging.info("训练被手动中断，保存当前模型...")  
        torch.save(model.state_dict(), exp_dir / 'interrupted_model.pth')  
    except Exception as e:  
        logging.error(f"训练出错: {str(e)}")  
        raise  

    return best_miou, metrics_history  

def main():
    """主程序入口"""
    try:
        # 加载配置
        config = get_config()
        
        """设置日志配置"""
        setup_logging()
    
        # 初始化模型  
        model = segmentation_model_factory(
            model_type='UNetWithReCLIPResNet',
            model_name=config['model']['model_name'],  
            ckpt_path=config['paths']['model']['clip_ckpt'],  
            num_classes=config['dataset']['num_classes'],  
            dropout_rate=0.2,  
            use_aux_loss=True,  
            initial_features=128,
            device_ids=[2,3]
        ) 

        # 初始化训练组件
        device, exp_dir, optimizer, scheduler, criterion = init_training(config)

        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(
            image_dir= Path(config['paths']['data']['image_dir']),
            labels_dir=Path(config['paths']['data']['label_dir']),
            batch_size=config['training']['batch_size'],
            train_ratio=config['dataset']['train_val_split'],
            num_workers=config['dataset']['num_workers'],
        )

        if torch.cuda.device_count() > 1:  
            model = nn.DataParallel(model)  
            logging.info(f"使用 {torch.cuda.device_count()} 个GPU训练")  

        # 开始训练
        best_miou, metrics_history = train_loop(
            model, train_loader, val_loader, criterion,
            optimizer, scheduler, device, config, exp_dir
        )
        
        # 训练完成后的总结
        print("\n训练总结:")
        print(f"最佳mIoU: {best_miou:.4f}")
        
        # 保存最终的训练历史
        with open(exp_dir / 'final_metrics.json', 'w') as f:
            json.dump({
                'best_miou': float(best_miou),
                'metrics_history': metrics_history
            }, f, indent=4)

    except Exception as e:
        logging.error(f"程序出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()