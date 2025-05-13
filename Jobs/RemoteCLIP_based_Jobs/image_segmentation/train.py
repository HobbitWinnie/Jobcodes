import sys  
sys.path.append('/home/nw/Codes')  

import logging  
from datetime import datetime
from pathlib import Path  
import json  
import torch.optim as optim  
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  

from Models.RemoteCLIP_based_Segmentation.modules.combined_loss import CombinedLoss
from Models.RemoteCLIP_based_Segmentation.factory import segmentation_model_factory  
from engine.base_trainer import BaseTrainer  
from engine.trainer import Trainer
from config.config import get_config        
from utils.set_logging import setup_logging  
from data.dataset import create_dataloaders  


def path_to_str(obj):  
    if isinstance(obj, dict):  
        return {k: path_to_str(v) for k, v in obj.items()}  
    elif isinstance(obj, list):  
        return [path_to_str(i) for i in obj]  
    elif isinstance(obj, Path):  
        return str(obj)  
    else:  
        return obj  
    
def main():  
    # 读取参数
    config = get_config()  
    
    # 设置日志
    # exp_dir = Path(config['paths']['model']['save_dir'])  
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(__file__).parent/'model_save'/ timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)      
    
    with open(exp_dir / 'config.json', 'w') as f:  
        json.dump(path_to_str(config.config), f, indent=4)  
    setup_logging(exp_dir)  
 
    # 初始化模型  
    model = segmentation_model_factory(  
        model_type='ReCLIPViTSeg',   # 可选的还有：'UNetWithReCLIPResNet', ReCLIPResNetSeg, ReCLIPViTSeg, CLIPVITSegmentation
        model_name=config['model']['model_name'],  
        ckpt_path=config['paths']['model']['clip_ckpt'],  
        num_classes=config.dataset['num_classes'],  
        dropout_rate=0.2,  
        use_aux_loss=True,  
        initial_features=32,  
        device_ids=[3],
        in_channels = 4  
    )  

    #
    optimizer = optim.AdamW(  
        model.parameters(),  
        lr=config['training']['learning_rate'],  
        weight_decay=config['training']['weight_decay'],  
        betas=(0.9, 0.999)  
    )  
    scheduler = CosineAnnealingWarmRestarts(  
        optimizer,  
        T_0=config['training']['scheduler_T0'],  
        T_mult=config['training']['scheduler_T_mult'],  
        eta_min=config['training']['min_lr']  
    )  
    criterion = CombinedLoss(  
        gamma=config['training'].get('gamma', 2.0),  
        alpha=config['training'].get('alpha', 0.5),  
        ignore_index=config['training']['ignore_index']  
    )
    
    train_loader, val_loader = create_dataloaders(  
        image_dir=Path(config['paths']['data']['image_dir']),  
        labels_dir=Path(config['paths']['data']['label_dir']),  
        batch_size=config['training']['batch_size'],  
        train_ratio=config['dataset']['train_val_split'],  
        num_workers=config['dataset']['num_workers'],  
    )  

    # 训练  
    trainer = Trainer(model, optimizer, scheduler, criterion, exp_dir, config)  
    best_miou = trainer.train(train_loader, val_loader)  

    logging.info("\n训练总结:")  
    logging.info(f"最佳mIoU: {best_miou:.4f}")  

if __name__ == '__main__':  
    main()  