import os  
import torch  
import torch.nn as nn  
import logging  
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  
from torch.utils.data import DataLoader  
from torch.cuda.amp import GradScaler, autocast  
from torch.nn import functional as F  
from tqdm import tqdm  
import numpy as np  

from config import get_config, setup_logging, setup_device  
from utils import (  
    RomoClipLoss,   
    EarlyStopping,   
    check_grad_norm,  
    load_and_save_data,  
    calculate_iou,  
    multiclass_dice_coefficient  
)  
from dataset import create_dataloaders  
from model import RomoClipUNet  

class CLIPFeatureLoss(nn.Module):  
    """CLIP特征损失"""  
    def __init__(self):  
        super().__init__()  
        self.mse = nn.MSELoss()  
        
    def forward(self, pred_features, target_features):  
        return self.mse(pred_features, target_features)  

def setup_criterion(config):  
    """初始化损失函数"""  
    return {  
        'ce': nn.CrossEntropyLoss(ignore_index=config['training'].get('ignore_index', 0)),  
        'dice': CombinedLoss(  
            weights=config['training'].get('loss_weights', [0.5, 0.5]),  
            ignore_index=config['training'].get('ignore_index', 0)  
        ),  
        'feature': CLIPFeatureLoss()  
    }  
def validate(model, val_loader, criterion, device):  
    """验证函数"""  
    model.eval()  
    total_loss = 0  
    loss_components = {'ce_loss': 0, 'dice_loss': 0, 'feature_loss': 0}  
    
    with torch.no_grad():  
        for batch in val_loader:  
            images = batch['image'].to(device)  
            masks = batch['mask'].to(device)  
            clip_features = batch.get('clip_features')  
            if clip_features is not None:  
                clip_features = clip_features.to(device)  
            
            with autocast():  
                outputs = model(images)  
                loss, components = criterion(  
                    outputs=outputs,  
                    targets=masks,  
                    clip_features=clip_features,  
                    pred_features=outputs.get('features')  
                )  
            
            total_loss += loss.item()  
            for k, v in components.items():  
                loss_components[k] += v  
    
    # 计算平均值  
    num_batches = len(val_loader)  
    return {  
        'val_loss': total_loss / num_batches,  
        **{k: v / num_batches for k, v in loss_components.items()}  
    }  

def train_model(model, train_loader, val_loader, device, config):  
    """训练模型的主函数"""  
    logging.info("开始训练...")  
    
    model = model.to(device)  
    criterion = RomoClipLoss(  
        weights=config['training'].get('loss_weights', [0.5, 0.3, 0.2]),  
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
        eta_min=config['training']['min_lr']  
    )  
    
    scaler = GradScaler()  
    early_stopping = EarlyStopping(patience=config['training']['patience'])  
    best_val_loss = float('inf')  
    
    for epoch in range(config['training']['epochs']):  
        model.train()  
        epoch_loss = 0  
        loss_components = {'ce_loss': 0, 'dice_loss': 0, 'feature_loss': 0}  
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):  
            images = batch['image'].to(device)  
            masks = batch['mask'].to(device)  
            clip_features = batch.get('clip_features')  
            if clip_features is not None:  
                clip_features = clip_features.to(device)  
            
            optimizer.zero_grad()  
            
            try:  
                with autocast():  
                    outputs = model(images)  
                    loss, components = criterion(  
                        outputs=outputs,  
                        targets=masks,  
                        clip_features=clip_features,  
                        pred_features=outputs.get('features')  
                    )  
                
                scaler.scale(loss).backward()  
                
                if config['training'].get('max_grad_norm'):  
                    torch.nn.utils.clip_grad_norm_(  
                        model.parameters(),  
                        config['training']['max_grad_norm']  
                    )  
                
                scaler.step(optimizer)  
                scaler.update()  
                
                epoch_loss += loss.item()  
                for k, v in components.items():  
                    loss_components[k] += v  
                
            except RuntimeError as e:  
                logging.error(f"训练批次错误: {str(e)}")  
                continue  
        
        # 计算平均损失  
        num_batches = len(train_loader)  
        train_stats = {  
            'train_loss': epoch_loss / num_batches,  
            **{k: v / num_batches for k, v in loss_components.items()}  
        }  
        
        # 验证  
        val_stats = validate(model, val_loader, criterion, device)  
        
        scheduler.step()  
        current_lr = optimizer.param_groups[0]['lr']  
        
        # 记录训练信息  
        logging.info(  
            f"Epoch [{epoch + 1}/{config['training']['epochs']}] "  
            f"Train Loss: {train_stats['train_loss']:.4f} "  
            f"(CE: {train_stats['ce_loss']:.4f}, "  
            f"Dice: {train_stats['dice_loss']:.4f}, "  
            f"Feature: {train_stats['feature_loss']:.4f}) "  
            f"Val Loss: {val_stats['val_loss']:.4f} "  
            f"LR: {current_lr:.6f}"  
        )  
        
        # 保存最佳模型  
        if val_stats['val_loss'] < best_val_loss:  
            best_val_loss = val_stats['val_loss']  
            torch.save({  
                'epoch': epoch,  
                'model_state_dict': model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),  
                'scheduler_state_dict': scheduler.state_dict(),  
                'best_val_loss': best_val_loss,  
                'config': config  
            }, os.path.join(config['paths']['model']['save_dir'], 'best_model.pth'))  
            logging.info(f"保存最佳模型 (epoch {epoch + 1})")  
        
        if early_stopping(val_stats['val_loss']):  
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")  
            break  
            
    return best_val_loss  

def main():  
    """主函数"""  
    try:  
        # 加载配置  
        config = get_config()  
        
        # 设置日志  
        logging.basicConfig(  
            level=logging.INFO,  
            format='%(asctime)s - %(levelname)s - %(message)s',  
            handlers=[  
                logging.FileHandler(  
                    os.path.join(config['paths']['model']['save_dir'], 'training.log')  
                ),  
                logging.StreamHandler()  
            ]  
        )  
        
        # 设置设备  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        logging.info(f"使用设备: {device}")  
        
        # 创建数据加载器  
        train_loader, val_loader = create_dataloaders(config)  
        logging.info("数据加载器创建成功")  
        
        # 初始化模型  
        model = YourModel(**config['model'])  
        if torch.cuda.device_count() > 1:  
            model = nn.DataParallel(model)  
        logging.info(f"模型初始化完成，使用 {torch.cuda.device_count()} 个GPU")  
        
        # 训练模型  
        best_loss = train_model(  
            model=model,  
            train_loader=train_loader,  
            val_loader=val_loader,  
            device=device,  
            config=config  
        )  
        
        logging.info(f"训练完成! 最佳验证损失: {best_loss:.4f}")  
        
    except Exception as e:  
        logging.error(f"训练过程发生错误: {str(e)}")  
        raise  

if __name__ == "__main__":  
    main()  