import os  
import torch  
import torch.nn as nn  
import logging  
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  
from torch.utils.data import DataLoader  
from torch.cuda.amp import GradScaler, autocast  
from torch.nn import functional as F  

from config import get_config, setup_logging, setup_device  
from utils import (  
    CombinedLoss,   
    EarlyStopping,   
    check_grad_norm,  
    load_and_save_data,  
    calculate_iou,  
    multiclass_dice_coefficient  
)  
from dataset import RemoteSensingDataset, create_dataloaders  
from model import UNet  

def validate(model, val_loader, criterion, device):  
    """验证函数"""  
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
                loss, loss_dict = criterion(outputs, label_patch)  
                
                pred = F.softmax(outputs, dim=1)  
                
                val_stats['total_loss'] += loss.item()  
                val_stats['ce_loss'] += loss_dict['ce_loss']  
                val_stats['dice_loss'] += loss_dict['dice_loss']  
                
                dice_score = multiclass_dice_coefficient(  
                    pred,  
                    label_patch,  
                    (label_patch != criterion.ignore_index).float(),  
                    ignore_index=criterion.ignore_index  
                )  
                val_stats['dice_score'] += dice_score.item()  
                
                iou_score = calculate_iou(  
                    pred,  
                    label_patch,  
                    num_classes=outputs.size(1),  
                    ignore_index=criterion.ignore_index  
                )  
                val_stats['iou_score'] += iou_score.item()  
    
    num_batches = len(val_loader)  
    val_stats = {k: v / num_batches for k, v in val_stats.items()}  
    
    return val_stats  

def train_model(model, train_loader, val_loader, device, config):  
    """训练模型的主函数"""  
    logging.info("Starting training with configuration:")  
    for key, value in config['training'].items():  
        logging.info(f"{key}: {value}")  

    model = model.to(device)  
    criterion = CombinedLoss(  
        weights=config['training'].get('loss_weights', [0.5, 0.5]),  
        ignore_index=config['training'].get('ignore_index', 0)  
    )  
    
    optimizer = torch.optim.AdamW(  
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
    
    scaler = GradScaler()  
    early_stopping = EarlyStopping(patience=config['training']['patience'])  
    best_val_dice = 0  
    best_epoch = 0  
    
    for epoch in range(config['training']['epochs']):  
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
                
                if check_grad_norm(model) > config['training']['max_grad_norm']:  
                    torch.nn.utils.clip_grad_norm_(  
                        model.parameters(),  
                        config['training']['max_grad_norm']  
                    )  
                
                scaler.step(optimizer)  
                scaler.update()  
                
                train_stats['total_loss'] += loss.item()  
                train_stats['ce_loss'] += loss_dict['ce_loss']  
                train_stats['dice_loss'] += loss_dict['dice_loss']  

            except RuntimeError as e:  
                logging.error(f"Error in training batch {batch_idx}: {str(e)}")  
                continue  
        
        train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}  
        val_stats = validate(model, val_loader, criterion, device)  
        
        scheduler.step()  
        current_lr = optimizer.param_groups[0]['lr']  
        
        logging.info(  
            f"Epoch [{epoch + 1}/{config['training']['epochs']}] "  
            f"Train Loss: {train_stats['total_loss']:.4f} "  
            f"Val Loss: {val_stats['total_loss']:.4f} "  
            f"Val Dice: {val_stats['dice_score']:.4f} "  
            f"Val IoU: {val_stats['iou_score']:.4f} "  
            f"LR: {current_lr:.6f}"  
        )  
        
        if (val_stats['dice_score'] > best_val_dice) and epoch > 100:  
            best_val_dice = val_stats['dice_score']  
            best_epoch = epoch  
            save_path = config.get_model_path()  
            torch.save({  
                'epoch': epoch,  
                'model_state_dict': model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),  
                'best_val_dice': best_val_dice,  
                'config': config.config,  
                'val_stats': val_stats  
            }, save_path)  
            logging.info(f"New best model saved at epoch {epoch + 1}")  
        
        if early_stopping(val_stats['total_loss']):  
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")  
            break  
    
    return best_val_dice  

def main():  
    try:  
        # 获取配置并初始化  
        config = get_config()  
        config.create_directories()  
        
        # 设置日志  
        log_file = os.path.join(config['paths']['model']['save_dir'], 'training.log')  
        setup_logging(log_file)  
        
        # 设置设备  
        device = setup_device()  
        
        # 构建路径  
        image_path = os.path.join(  
            config['paths']['data']['images'],  
            config['paths']['input']['train_image']  
        )  
        label_path = os.path.join(  
            config['paths']['data']['images'],  
            config['paths']['input']['train_label']  
        )  
        process_dir = config['paths']['data']['process']  
        
        # 加载和处理数据  
        image, labels, _ = load_and_save_data(  
            image_path=image_path,  
            label_path=label_path,  
            output_dir=process_dir,  
            normalize=True  
        )  
        logging.info(f"Data loaded successfully - Image shape: {image.shape}, Labels shape: {labels.shape}")  
        
        # 创建数据加载器  
        train_loader, val_loader = create_dataloaders(  
            image=image,  
            labels=labels,  
            patch_size=config['dataset']['patch_size'],  
            num_patches=config['dataset']['patch_number'],  
            batch_size=config['training']['batch_size'],  
            train_ratio=config['dataset']['train_val_split']  
        )  
        logging.info("Dataloaders created successfully")  

        # 初始化模型  
        model = UNet(**config['model'])  
        if torch.cuda.device_count() > 1:  
            model = nn.DataParallel(model)  
        logging.info(f"Model initialized with {torch.cuda.device_count()} GPUs")  
        
        # 训练模型  
        best_dice = train_model(  
            model=model,  
            train_loader=train_loader,  
            val_loader=val_loader,  
            device=device,  
            config=config  
        )  
        
        logging.info(f"Training completed with best validation Dice score: {best_dice:.4f}")  
        
    except Exception as e:  
        logging.error(f"Error during training: {str(e)}")  
        raise  

if __name__ == "__main__":  
    main()