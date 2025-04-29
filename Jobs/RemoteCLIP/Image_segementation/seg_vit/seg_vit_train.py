
import sys
sys.path.append('/home/nw/Codes/RemoteCLIP/Image_segementation')  

import torch
import torch.optim as optim
import logging
import time
import numpy as np
import json
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from data.dataset import create_dataloaders
from seg_vit_model import CLIPSegmentation
from config import get_config
from combined_loss import CombinedLoss
from utils import setup_logging


def init_training(config):
    """初始化训练组件"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logging.info(f"CUDA版本: {torch.version.cuda}")
        logging.info(f"可用GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f}MB")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(__file__).parent/'model_save'/ timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(exp_dir / 'training.log')

    # 保存配置文件  
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config.config, f, indent=4)

    # 初始化模型
    num_classes = config['dataset']['num_classes']
    model = CLIPSegmentation(
        model_name='ViT-L-14',  # 指定使用 ViT-L-14 模型
        ckpt_path='/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-L-14.pt',  # 如果有预训练权重，可在此指定
        num_classes=num_classes,  # 分割任务类别数
        input_size=config['dataset']['patch_size'],  # 输入图像大小，应与 ViT-L-14 模型匹配
        freeze_clip=False  # 解冻 CLIP 模型参数
    ).to(device)
    logging.info("模型初始化完成")  

    # 初始化优化器  
    optimizer = optim.AdamW(  
        filter(lambda p: p.requires_grad, model.parameters()),  
        lr=config['training']['learning_rate'],  
        weight_decay=config['training']['weight_decay'],  
        betas=(0.9, 0.999)  
    )
    logging.info(f"优化器: AdamW, 初始学习率: {config['training']['learning_rate']}")  

    # # 初始化学习率调度器
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=5,
    #     min_lr=1e-7,
    #     verbose=True
    # )
    
    # 使用 CosineAnnealingWarmRestarts 学习率调度器  
    scheduler = CosineAnnealingWarmRestarts(  
        optimizer,  
        T_0=config['training'].get('scheduler_T_0', 10),       # 初始周期  
        T_mult=config['training'].get('scheduler_T_mult', 2),  # 周期倍率  
        eta_min=config['training'].get('scheduler_eta_min', 1e-7)  # 最小学习率  
    )  
    logging.info("学习率调度器: CosineAnnealingWarmRestarts")  

    # 初始化损失函数
    criterion = CombinedLoss(
        gamma=config['training'].get('gamma', 2.0),
        alpha=config['training'].get('alpha', 0.5),
        ignore_index=config['training']['ignore_index']
    ).to(device)

    return device, exp_dir, model, optimizer, scheduler, criterion

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

            preds = outputs.argmax(1)

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
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_focal_loss += loss_info['focal_loss']
                epoch_dice_loss += loss_info['dice_loss']
                batch_count += 1

                del outputs, loss
                torch.cuda.empty_cache()

            avg_loss = epoch_loss / batch_count
            avg_focal_loss = epoch_focal_loss / batch_count
            avg_dice_loss = epoch_dice_loss / batch_count
            epoch_time = time.time() - epoch_start


            metrics_history['train_loss'].append(avg_loss)
            metrics_history['train_focal_loss'].append(avg_focal_loss)
            metrics_history['train_dice_loss'].append(avg_dice_loss)
            metrics_history['learning_rate'].append(current_lr)

            val_metrics = validate_model(   
                model, val_loader, criterion, device,
                config['dataset']['num_classes']
            )

            # scheduler.step(val_metrics['loss'])

            # 更新学习率调度器（对于CosineAnnealingWarmRestarts，可以在每个batch或epoch后更新）  
            scheduler.step()  
        
            metrics_history['val_loss'].append(val_metrics['loss'])
            metrics_history['val_focal_loss'].append(val_metrics['focal_loss'])
            metrics_history['val_dice_loss'].append(val_metrics['dice_loss'])
            metrics_history['val_miou'].append(val_metrics['mean_iou'])
            metrics_history['val_accuracy'].append(val_metrics['accuracy'])

            logging.info(
                f"Epoch {epoch + 1}/{total_epochs} [{epoch_time:.2f}s], "
                f"Training: [{avg_loss:.4f} (Focal: {avg_focal_loss:.4f}, Dice: {avg_dice_loss:.4f})], "
                f"Validation: [{val_metrics['loss']:.4f} (Focal: {val_metrics['focal_loss']:.4f}, Dice: {val_metrics['dice_loss']:.4f}), "
                f"Acc = {val_metrics['accuracy']:.4f}, mIoU = {val_metrics['mean_iou']:.4f}], LR: {current_lr:.6f}"
            )

            if val_metrics['mean_iou'] > best_miou:
                best_miou = val_metrics['mean_iou']
                torch.save(
                    model.state_dict(), exp_dir / 'best_model.pth'
                )
                logging.info(f"保存最佳模型 (mIoU: {best_miou:.4f})")

    except KeyboardInterrupt:
        logging.info("训练被手动中断，保存当前模型...")
    except Exception as e:
        logging.error(f"训练出错: {str(e)}")
        raise

    return best_miou, metrics_history


def main():
    try:
        config = get_config()
        device, exp_dir, model, optimizer, scheduler, criterion = init_training(config)

        train_loader, val_loader = create_dataloaders(
            image_dir=Path(config['paths']['data']['image_dir']),
            labels_dir=Path(config['paths']['data']['label_dir']),
            batch_size=config['training']['batch_size'],
            train_ratio=config['dataset']['train_val_split'],
            num_workers=config['dataset']['num_workers'],
       )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logging.info(f"使用 {torch.cuda.device_count()} 个GPU训练")

        best_miou, metrics_history = train_loop(
            model, train_loader, val_loader, criterion,
            optimizer, scheduler, device, config, exp_dir
        )

        print("\n训练总结:")
        print(f"最佳mIoU: {best_miou:.4f}")

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

