import logging  
from pathlib import Path  
import torch  
import torch.optim as optim  
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  

from Models.RemoteCLIP_based_Segmentation.modules.combined_loss import CombinedLoss
from Models.RemoteCLIP_based_Segmentation.factory import segmentation_model_factory  
from engine.reclip_rn50_unet_train import UNetTrainer  
from config.config import get_config        
from utils.set_logging import setup_logging  
from data.dataset import create_dataloaders  


def init_training(config, model):  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    if torch.cuda.is_available():  
        torch.backends.cudnn.benchmark = True  
        torch.backends.cudnn.deterministic = False  
        logging.info(f"CUDA版本: {torch.version.cuda}")  
        logging.info(f"可用GPU: {torch.cuda.get_device_name(0)}")  
        logging.info(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")  

    exp_dir = Path(config['paths']['model']['save_dir'])  
    exp_dir.mkdir(parents=True, exist_ok=True)  

    setup_logging(exp_dir)  
    import json  
    with open(exp_dir / 'config.json', 'w') as f:  
        json.dump(config.config, f, indent=4)  

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
    ).to(device)  
    return device, exp_dir, optimizer, scheduler, criterion  

def main():  
    config = get_config()  
    setup_logging()  
    # 初始化模型  
    model = segmentation_model_factory(  
        model_type='UNetWithReCLIPResNet',  
        model_name=config['model']['model_name'],  
        ckpt_path=config['paths']['model']['clip_ckpt'],  
        num_classes=config.dataset['num_classes'],  
        dropout_rate=0.2,  
        use_aux_loss=True,  
        initial_features=128,  
        device_ids=[2, 3]  
    )  

    device, exp_dir, optimizer, scheduler, criterion = init_training(config, model)  
    train_loader, val_loader = create_dataloaders(  
        image_dir=Path(config['paths']['data']['image_dir']),  
        labels_dir=Path(config['paths']['data']['label_dir']),  
        batch_size=config['training']['batch_size'],  
        train_ratio=config['dataset']['train_val_split'],  
        num_workers=config['dataset']['num_workers'],  
    )  

    # 训练  
    trainer = UNetTrainer(model, optimizer, scheduler, criterion, device, exp_dir, config)  
    best_miou, metrics_history = trainer.train(train_loader, val_loader)  

    print("\n训练总结:")  
    print(f"最佳mIoU: {best_miou:.4f}")  
    import json  
    with open(exp_dir / 'final_metrics.json', 'w') as f:  
        json.dump({'best_miou': float(best_miou), 'metrics_history': metrics_history}, f, indent=4)  

if __name__ == '__main__':  
    main()  