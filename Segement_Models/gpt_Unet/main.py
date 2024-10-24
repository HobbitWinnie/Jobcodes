import os  
import torch  
import rasterio  
import torch.nn as nn  
import logging  
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from torch.utils.data import DataLoader  
from torch.cuda.amp import GradScaler, autocast  
from tqdm import tqdm  
from torch.nn import functional as F
from datetime import datetime  

from dataset import RemoteSensingDataset, reconstruct_image_from_patches, split_image_into_patches  
from utils import load_data, mean_iou, multiclass_dice_coefficient
from model import UNet  

# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  


# 使用组合损失函数  
class CombinedLoss(nn.Module):  
    def __init__(self, weights=[0.5, 0.5]):  
        super().__init__()  
        self.ce = nn.CrossEntropyLoss(reduction='none')  
        self.weights = weights  
    def forward(self, outputs, targets, mask):  
        ce_loss = self.ce(outputs, targets)  
        dice_loss = 1 - multiclass_dice_coefficient(  
            F.softmax(outputs, dim=1),  
            targets,  
            mask  
        )  
        masked_ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-6)  
        return self.weights[0] * masked_ce_loss + self.weights[1] * dice_loss  


class EarlyStopping:  
    def __init__(self, patience=100, min_delta=1e-4):  
        self.patience = patience  
        self.min_delta = min_delta  
        self.counter = 0  
        self.best_loss = None  
        self.should_stop = False  

    def __call__(self, val_loss):  
        if self.best_loss is None:  
            self.best_loss = val_loss  
        elif val_loss > self.best_loss - self.min_delta:  
            self.counter += 1  
            if self.counter >= self.patience:  
                self.should_stop = True  
        else:  
            self.best_loss = val_loss  
            self.counter = 0  

def setup_logging(save_dir):  
    """设置日志记录"""  
    os.makedirs(save_dir, exist_ok=True)  
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')  
    logging.basicConfig(  
        level=logging.INFO,  
        format='%(asctime)s [%(levelname)s] %(message)s',  
        handlers=[  
            logging.FileHandler(log_file),  
            logging.StreamHandler()  
        ]  
    )  

def check_grad_norm(model):  
    """检查梯度范数"""  
    total_norm = 0  
    for p in model.parameters():  
        if p.grad is not None:  
            param_norm = p.grad.data.norm(2)  
            total_norm += param_norm.item() ** 2  
    total_norm = total_norm ** (1. / 2)  
    return total_norm  

def train(  
    model,   
    train_loader,   
    val_loader,   
    device,   
    config  
):  
    """  
    改进的训练函数  
    
    Args:  
        model: 模型实例  
        train_loader: 训练数据加载器  
        val_loader: 验证数据加载器  
        device: 训练设备  
        config: 配置字典，包含所有超参数  
    """  
    # 设置日志  
    setup_logging(config['save_dir'])  
    logging.info(f"Starting training with config: {config}")  
    
    model = model.to(device)  
    criterion = CombinedLoss()  
    
    # 优化器设置  
    optimizer = torch.optim.AdamW(  
        model.parameters(),  
        lr=config['learning_rate'],  
        weight_decay=config['weight_decay'],  
        eps=1e-8  
    )  
    
    # 学习率调度器  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(  
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
            'loss': 0,  
            'dice': 0,  
            'iou': 0  
        }  
        
        for batch_idx, batch in enumerate(train_loader):  
            img_patch, label_patch, mask_patch = [b.to(device) for b in batch]  
            
            # 检查输入数据  
            if torch.isnan(img_patch).any():  
                logging.warning(f"NaN detected in input data at epoch {epoch}, batch {batch_idx}")  
                continue  
                
            optimizer.zero_grad()  
            
            try:  
                with autocast():  
                    outputs = model(img_patch)  
                    pixel_wise_loss = criterion(outputs, label_patch.long(), mask_patch)  
                    
                    # 添加数值稳定性检查  
                    mask_sum = mask_patch.sum() + 1e-8  
                    if mask_sum < 1e-7:  
                        logging.warning(f"Very small mask sum detected: {mask_sum}")  
                        continue  
                        
                    masked_loss = (pixel_wise_loss * mask_patch).sum() / mask_sum  
                    
                    if torch.isnan(masked_loss):  
                        logging.error(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")  
                        continue  
                
                scaler.scale(masked_loss).backward()  
                
                # 梯度裁剪  
                grad_norm = check_grad_norm(model)  
                if grad_norm > config['max_grad_norm']:  
                    logging.warning(f"Large gradient detected: {grad_norm}")  
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])  
                
                scaler.step(optimizer)  
                scaler.update()  
                
                # 收集训练统计  
                with torch.no_grad():  
                    pred = F.softmax(outputs, dim=1)  
                    train_stats['loss'] += masked_loss.item()  
                    train_stats['dice'] += multiclass_dice_coefficient(pred, label_patch, mask_patch).item()  
                    train_stats['iou'] += mean_iou(pred, label_patch, mask_patch).item()  
                
            except RuntimeError as e:  
                logging.error(f"Error in training batch: {str(e)}")  
                continue  
        
        # 计算平均训练指标  
        train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}  
        
        # 验证阶段  
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)  
        
        # 学习率调整  
        current_lr = optimizer.param_groups[0]['lr']  
        scheduler.step()  
        
        # 记录训练信息  
        logging.info(  
            f"Epoch [{epoch + 1}/{config['epochs']}] "  
            f"Train Loss: {train_stats['loss']:.4f} "  
            f"Train Dice: {train_stats['dice']:.4f} "  
            f"Train IoU: {train_stats['iou']:.4f} "  
            f"Val Loss: {val_loss:.4f} "  
            f"Val Dice: {val_dice:.4f} "  
            f"Val IoU: {val_iou:.4f} "  
            f"LR: {current_lr:.6f}"  
        )  
        
        # 模型保存  
        if val_dice > best_val_dice:  
            best_val_dice = val_dice  
            best_epoch = epoch  
            save_path = os.path.join(config['save_dir'], 'best_model.pth')  
            torch.save({  
                'epoch': epoch,  
                'model_state_dict': model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),  
                'scheduler_state_dict': scheduler.state_dict(),  
                'best_val_dice': best_val_dice,  
                'config': config  
            }, save_path)  
            logging.info(f"New best model saved at epoch {epoch + 1}")  
        
        # 每10个epoch记录模型参数统计  
        if epoch % 10 == 0:  
            for name, param in model.named_parameters():  
                if param.requires_grad:  
                    logging.info(  
                        f"Layer {name}: "  
                        f"mean={param.data.mean().item():.4f}, "  
                        f"std={param.data.std().item():.4f}, "  
                        f"grad_mean={param.grad.mean().item():.4f} if param.grad is not None else 'None'"  
                    )  
        
        # 早停检查  
        early_stopping(val_loss)  
        if early_stopping.should_stop:  
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")  
            break  
    
    logging.info(f"Training completed. Best Val Dice: {best_val_dice:.4f} at epoch {best_epoch + 1}")  
    return best_val_dice  

def validate(model, val_loader, criterion, device):  
    """验证函数"""  
    model.eval()  
    val_stats = {  
        'loss': 0,  
        'dice': 0,  
        'iou': 0  
    }  
    
    with torch.no_grad():  
        for batch in val_loader:  
            img_patch, label_patch, mask_patch = [b.to(device) for b in batch]  
            
            with autocast():  
                outputs = model(img_patch)  
                pixel_wise_loss = criterion(outputs, label_patch.long(), mask_patch)  
                masked_loss = (pixel_wise_loss * mask_patch).sum() / (mask_patch.sum() + 1e-8)  
            
            pred = F.softmax(outputs, dim=1)  
            val_stats['loss'] += masked_loss.item()  
            val_stats['dice'] += multiclass_dice_coefficient(pred, label_patch, mask_patch).item()  
            val_stats['iou'] += mean_iou(pred, label_patch, mask_patch).item()  
    
    # 计算平均值  
    val_stats = {k: v / len(val_loader) for k, v in val_stats.items()}  
    
    return val_stats['loss'], val_stats['dice'], val_stats['iou']  

def predict(model, save_path, test_image_paths, output_paths, patch_size, overlap, device):  
    model.load_state_dict(torch.load(save_path, map_location=device))  
    model.eval()  

    for test_image_path, output_path in zip(test_image_paths, output_paths):  
        logging.info(f"Predicting for {test_image_path}")  
        test_image, _, image_profile = load_data(test_image_path)# image, labels, image_meta  
        patches = split_image_into_patches(test_image, patch_size, overlap)  
        predictions = []  

        with torch.no_grad():  
            for patch in tqdm(patches, desc="Processing patches"):  
                patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  
                output = model(patch)  
                pred = F.softmax(output, dim=1).squeeze().cpu().numpy()  
                predictions.append(pred)  

        reconstructed_prediction = reconstruct_image_from_patches(predictions, test_image.shape, patch_size, overlap)  

        #更新图像配置  
        image_profile.update(dtype=rasterio.uint8, count=1, nodata=0)  

        with rasterio.open(output_path, 'w', **image_profile) as dst:  
            # 写入最终预测  
            dst.write(reconstructed_prediction.astype(rasterio.uint8), 1)  

        logging.info(f"Prediction saved to {output_path}")  


def main():  
    # 配置字典  
    config = {  
        # 路径配置  
        'paths': {  
            'image_root': '/home/Dataset/nw/Segmentation/CpeosTest/images',  
            'save_dir': '/home/nw/Codes/Segement_Models/model_save',  
            'log_dir': '/home/nw/Codes/Segement_Models/logs',  
            'result_dir': '/home/Dataset/nw/Segmentation/CpeosTest/result'  
        },  
        
        # 数据集配置  
        'dataset': {  
            'patch_size': 256,  
            'patch_number': 5000,  
            'overlap': 64,  
            'train_val_split': 0.8  
        },  
        
        # 训练配置  
        'training': {  
            'epochs': 1000,  
            'batch_size': 128,  
            'learning_rate': 5e-4,  
            'min_lr': 1e-6,  
            'weight_decay': 0.001,  
            'scheduler_T0': 30,  
            'scheduler_T_mult': 2,  
            'patience': 100,  
            'max_grad_norm': 1.0  
        },  
        
        # 模型配置  
        'model': {  
            'in_channels': 4,  
            'out_channels': 10,  
            'initial_features': 64,  
            'dropout_rate': 0.2  
        }  
    }  
    
    # 创建必要的目录  
    for dir_path in [config['paths']['save_dir'], config['paths']['log_dir'], config['paths']['result_dir']]:  
        os.makedirs(dir_path, exist_ok=True)  
    
    # 设置日志  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    log_file = os.path.join(config['paths']['log_dir'], f'training_{timestamp}.log')  
    logging.basicConfig(  
        level=logging.INFO,  
        format='%(asctime)s [%(levelname)s] %(message)s',  
        handlers=[  
            logging.FileHandler(log_file),  
            logging.StreamHandler()  
        ]  
    )  
    
    # 记录配置信息  
    logging.info("Starting training with configuration:")  
    logging.info(config)  
    
    # 设置设备  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logging.info(f"Using device: {device}")  
    
    # 构建文件路径  
    image_path = os.path.join(config['paths']['image_root'], 'GF2_train_image.tif')  
    label_path = os.path.join(config['paths']['image_root'], 'train_label.tif')  
    test_img_paths = [  
        os.path.join(config['paths']['image_root'], 'train_mask.tif'),  
        os.path.join(config['paths']['image_root'], 'GF2_test_image.tif')  
    ]  
    output_paths = [  
        os.path.join(config['paths']['result_dir'], 'train_mask_gptUnet_results.tif'),  
        os.path.join(config['paths']['result_dir'], 'GF2_test_image_gptUnet_results.tif')  
    ]  
    
    # 初始化模型  
    try:  
        logging.info("Initializing UNet model")  
        model = UNet(  
            in_channels=config['model']['in_channels'],  
            out_channels=config['model']['out_channels'],  
            dropout_rate=config['model']['dropout_rate'],  
            initial_features=config['model']['initial_features']  
        )  
        
        if torch.cuda.device_count() > 1:  
            logging.info(f"Using {torch.cuda.device_count()} GPUs")  
            model = nn.DataParallel(model)  
        model.to(device)  
    except Exception as e:  
        logging.error(f"Error initializing model: {e}")  
        return  
    
    # 加载数据  
    try:  
        logging.info("Loading dataset")  
        image, labels, _ = load_data(image_path, label_path)  
        
        dataset = RemoteSensingDataset(  
            image,   
            labels,   
            patch_size=config['dataset']['patch_size'],  
            num_patches=config['dataset']['patch_number']  
        )  
        
        # 分割训练集和验证集  
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
        
        logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")  
        
    except Exception as e:  
        logging.error(f"Error preparing dataset: {e}")  
        return  
    
    # 开始训练  
    try:  
        logging.info("Starting training")  
        save_path = os.path.join(config['paths']['save_dir'], f'model_gptUNet_{timestamp}.pth')  
        best_dice = train(  
            model=model,  
            train_loader=train_loader,  
            val_loader=val_loader,  
            device=device,  
            config={  
                **config['training'],  
                'save_dir': config['paths']['save_dir']  
            }  
        )  
        logging.info(f"Training completed with best validation Dice score: {best_dice:.4f}")  
        
    except Exception as e:  
        logging.error(f"Error during training: {e}")  
        return  
    
    # 预测  
    try:  
        logging.info("Starting prediction")  
        predict(  
            model=model,  
            save_path=save_path,  
            test_img_paths=test_img_paths,  
            output_paths=output_paths,  
            patch_size=config['dataset']['patch_size'],  
            overlap=config['dataset']['overlap'],  
            device=device  
        )  
        logging.info("Prediction completed")  
        
    except Exception as e:  
        logging.error(f"Error during prediction: {e}")  
        return  
    
    logging.info("All processes completed successfully")  

if __name__ == "__main__":  
    main()  