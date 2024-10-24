import os
import torch
import torch.nn as nn
import logging
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from utils import (
    setup_logging, 
    CombinedLoss, 
    EarlyStopping, 
    check_grad_norm,
    load_and_save_data,
    calculate_iou,
    multiclass_dice_coefficient
)
from dataset import RemoteSensingDataset
from model import UNet

def validate(model, val_loader, criterion, device):  
    """  
    验证函数  
    
    Args:  
        model: 模型  
        val_loader: 验证数据加载器  
        criterion: 损失函数
        device: 计算设备  
    """  
    model.eval()  
    val_stats = {  
        'total_loss': 0,  
        'ce_loss': 0,  
        'dice_loss': 0,  
        'dice_score': 0,  
        'iou_score': 0  
    }  
    
    with torch.no_grad():  
        for batch in val_loader:  
            img_patch, label_patch = [b.to(device) for b in batch]  
            
            with autocast():  
                outputs = model(img_patch)  
                # 使用新的损失函数计算方式  
                loss, loss_dict = criterion(outputs, label_patch)  
                
                # 计算预测结果  
                pred = F.softmax(outputs, dim=1)  
                
                # 更新统计信息  
                val_stats['total_loss'] += loss.item()  
                val_stats['ce_loss'] += loss_dict['ce_loss']  
                val_stats['dice_loss'] += loss_dict['dice_loss']  
                
                # 计算每个批次的Dice分数  
                dice_score = multiclass_dice_coefficient(  
                    pred,   
                    label_patch,  
                    (label_patch != criterion.ignore_index).float(),  
                    ignore_index=criterion.ignore_index  
                )  
                val_stats['dice_score'] += dice_score.item()  
                
                # 计算IoU分数
                iou_score = calculate_iou(  
                    pred,  
                    label_patch,  
                    num_classes=outputs.size(1),  
                    ignore_index=criterion.ignore_index  
                )  
                val_stats['iou_score'] += iou_score.item()  
    
    # 计算平均值  
    num_batches = len(val_loader)  
    val_stats = {k: v / num_batches for k, v in val_stats.items()}  
    
    return val_stats  

def train_model(model, train_loader, val_loader, device, config):  
    """训练模型的主函数"""  
    setup_logging(config['save_dir'])  
    logging.info(f"Starting training with config: {config}")  
    
    model = model.to(device)  
    # 初始化新的损失函数  
    criterion = CombinedLoss(  
        weights=config.get('loss_weights', [0.5, 0.5]),  
        ignore_index=config.get('ignore_index', 0),
    )  
    
    optimizer = torch.optim.AdamW(  
        model.parameters(),  
        lr=config['learning_rate'],  
        weight_decay=config['weight_decay']  
    )  
    
    scheduler = CosineAnnealingWarmRestarts(  
        optimizer,  
        T_0=config['scheduler_T0'],  
        T_mult=config['scheduler_T_mult'],  
        eta_min=config['min_lr']  
    )  
    
    scaler = GradScaler()  
    early_stopping = EarlyStopping(patience=config['patience'])  
    best_val_dice = 0  
    best_epoch = 0  
    
    for epoch in range(config['epochs']):  
        # 训练阶段  
        model.train()  
        train_stats = {  
            'total_loss': 0,  
            'ce_loss': 0,  
            'dice_loss': 0,  
            'dice_score': 0  
        }  
        
        for batch_idx, batch in enumerate(train_loader):  
            img_patch, label_patch = batch   
            img_patch = img_patch.to(device)  
            label_patch = label_patch.to(device)  

            optimizer.zero_grad()  
           
            try:            
                with autocast():  
                    outputs = model(img_patch)  
                    loss, loss_dict = criterion(outputs, label_patch)  
                
                scaler.scale(loss).backward()  
                
                # 梯度裁剪  
                if check_grad_norm(model) > config['max_grad_norm']:  
                    torch.nn.utils.clip_grad_norm_(  
                        model.parameters(),  
                        config['max_grad_norm']  
                    )  
                
                scaler.step(optimizer)  
                scaler.update()  
                
                # 更新训练统计  
                train_stats['total_loss'] += loss.item()  
                train_stats['ce_loss'] += loss_dict['ce_loss']  
                train_stats['dice_loss'] += loss_dict['dice_loss']  
                
                if batch_idx % config.get('log_interval', 10) == 0:  
                    logging.info(  
                        f"Epoch [{epoch + 1}/{config['epochs']}] "  
                        f"Batch [{batch_idx}/{len(train_loader)}] "  
                        f"Loss: {loss.item():.4f} "  
                        f"CE: {loss_dict['ce_loss']:.4f} "  
                        f"Dice: {loss_dict['dice_loss']:.4f}"  
                    )  
                
            except RuntimeError as e:  
                logging.error(f"Error in training batch: {str(e)}")  
                continue  
        
        # 计算平均训练指标  
        train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}  
        
        # 验证阶段  
        val_stats = validate(model, val_loader, criterion, device)  
        
        # 学习率调整  
        scheduler.step()  
        current_lr = optimizer.param_groups[0]['lr']  
        
        # 记录训练信息  
        logging.info(  
            f"Epoch [{epoch + 1}/{config['epochs']}] "  
            f"Train Loss: {train_stats['total_loss']:.4f} "  
            f"Val Loss: {val_stats['total_loss']:.4f} "  
            f"Val Dice: {val_stats['dice_score']:.4f} "  
            f"Val IoU: {val_stats['iou_score']:.4f} "  
            f"LR: {current_lr:.6f}"  
        )  
        
        # 保存最佳模型  
        if val_stats['dice_score'] > best_val_dice:  
            best_val_dice = val_stats['dice_score']  
            best_epoch = epoch  
            save_path = os.path.join(config['save_dir'], 'best_model.pth')  
            torch.save({  
                'epoch': epoch,  
                'model_state_dict': model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),  
                'best_val_dice': best_val_dice,  
                'config': config,  
                'val_stats': val_stats  
            }, save_path)  
            logging.info(f"New best model saved at epoch {epoch + 1}")  
        
        # 早停检查  
        early_stopping(val_stats['total_loss'])  
        if early_stopping.should_stop:  
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")  
            break  
    
    return best_val_dice  



def main():
    # 加载配置
    config = {
        'paths': {
            'image_root': '/home/Dataset/nw/Segmentation/CpeosTest/images',
            'save_dir': '/home/nw/Codes/Segement_Models/model_save',
            'process_dir': '/home/Dataset/nw/Segmentation/CpeosTest'
        },
        'dataset': {
            'patch_size': 256,
            'patch_number': 5000,
            'train_val_split': 0.8,
            'num_classes': 9  # 添加类别数量配置  

        },
        'training': {
            'epochs': 1000,
            'batch_size': 128,
            'learning_rate': 5e-4,
            'min_lr': 1e-6,
            'weight_decay': 0.001,
            'scheduler_T0': 30,
            'scheduler_T_mult': 2,
            'patience': 100,
            'max_grad_norm': 1.0,
            'loss_weights': [0.5, 0.5],  # CE损失和Dice损失的权重  
            'ignore_index': 0,  # 设置忽略的标签值  
            'save_dir': '/home/nw/Codes/Segement_Models/model_save'
        },
        'model': {
            'in_channels': 4,
            'out_channels': 9,
            'initial_features': 64,
            'dropout_rate': 0.2
        }
    }
    
    # 创建保存目录
    os.makedirs(config['paths']['process_dir'], exist_ok=True)
    os.makedirs(config['paths']['save_dir'], exist_ok=True)

    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    image_path = os.path.join(config['paths']['image_root'], 'GF2_train_image.tif')
    label_path = os.path.join(config['paths']['image_root'], 'train_label.tif')
    output_dir = os.path.join(config['paths']['process_dir'],'image_process')

    
    try:
        image, labels, _ = load_and_save_data(
            image_path=image_path,
            label_path=label_path,
            output_dir=output_dir,
            normalize=True,
        )
        logging.info("Data loading and processing successful")

        dataset = RemoteSensingDataset(
            image, 
            labels, 
            patch_size=config['dataset']['patch_size'],
            num_patches=config['dataset']['patch_number']
        )
        
        # 分割数据集
        train_size = int(config['dataset']['train_val_split'] * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=16,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=16,
            pin_memory=True
        )
        
        # 初始化模型
        model = UNet(**config['model'])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # 训练模型
        best_dice = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config['training']
        )
        
        logging.info(f"Training completed with best validation Dice score: {best_dice:.4f}")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()