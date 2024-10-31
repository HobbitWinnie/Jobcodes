import os  
import yaml  
from typing import Dict, Any
import logging  
import torch  
import datetime  
from pathlib import Path  

class Config:  
    """配置管理类"""  
    
    VERSION = "1.0.0"  

    def __init__(self):  
        """  
        初始化配置  
        Args:  
            config_path: 配置文件路径,默认为None使用默认配置  
        """  
        self.config = self._get_default_config()  
        
        # 创建必要的目录  
        self.create_directories()  

        # 验证路径  
        self.validate_paths()  
            
    def _get_default_config(self) -> Dict[str, Any]:  
        """获取默认配置"""  
        return {  
            'paths': {  
                'data': {  
                    'root': '/home/Dataset/nw/Segmentation/CpeosTest',  
                    'images': '/home/Dataset/nw/Segmentation/CpeosTest/images',  
                    'process': '/home/Dataset/nw/Segmentation/CpeosTest/image_process',  
                    'results': '/home/Dataset/nw/Segmentation/CpeosTest/result'  
                },  
                'model': {  
                    'save_dir': '/home/nw/Codes/Segement_Models/model_save',  
                    'best_model': 'RomoClip_best_model.pth',  
                    'clip_model': {  
                        'path': '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-B-32.pt',  # 本地CLIP模型路径  
                        'version': 'ViT-B-32'  # 模型版本信息  
                    },  
                },  
                'input': {  
                    'train_image': 'GF2_train_image.tif',  
                    'train_label': 'train_label.tif',  
                    'train_mask': 'train_mask.tif',  
                    'test_image': 'GF2_test_image.tif'  
                },  
                'output': {  
                    'train_mask_result': 'train_mask_results_RomoClip.tif',  
                    'test_image_result': 'GF2_test_image_results_RomoClip.tif',  
                }  
            },  
            'dataset': {  
                'patch_size': 256,  
                'patch_number': 3000,  
                'train_val_split': 0.8,  
                'num_classes': 9,  
                'clip_input_size': 224,  
            },  
            'model': {  
                'in_channels': 4,  
                'out_channels': 9,  
                'initial_features': 128,  
                'dropout_rate': 0.2,  
                'feature_fusion': True,  
                'clip': {  
                    'embed_dim': 512,  
                    'feature_layers': [3, 6, 9],  
                    'freeze': True,  
                    'local_model': True,  
                    'input_resolution': 224,  
                    'patch_size': 32,  
                    'width': 768  
                },  
                'decoder': {  
                    'use_attention': True,  
                    'attention_type': 'self',  
                    'num_heads': 8  
                }  
            },  
            'training': {  
                'epochs': 2000,  
                'batch_size': 128,  
                'learning_rate': 1e-4,  
                'min_lr': 1e-6,  
                'weight_decay': 0.01,  
                'scheduler_T0': 30,  
                'scheduler_T_mult': 2,  
                'patience': 100,  
                'max_grad_norm': 1.0,  
                'loss_weights': {  
                    'ce_loss': 0.4,  
                    'dice_loss': 0.3,  
                    'feature_loss': 0.3  
                },  
                'ignore_index': 0,  
                'mixed_precision': True,  
                'gradient_accumulation_steps': 1,  
                'warmup_epochs': 5  
            },  
            'predict': {  
                'overlap': 64,  
                'batch_size': 16,  
                'tta': False,  
                'ensemble': False,  
                'post_processing': {  
                    'enable': True,  
                    'min_size': 100,  
                    'smoothing': True  
                }  
            },  
            'logging': {  
                'log_interval': 100,  
                'eval_interval': 1,  
                'save_interval': 5,  
                'tensorboard': True,  
                'wandb': {  
                    'enable': False,  
                    'project': 'romoclip',  
                    'name': None  
                }  
            }  
        }  
    
    def create_directories(self) -> None:  
        """创建必要的目录"""  
        directories = [  
            self.config['paths']['data']['process'],  
            self.config['paths']['data']['results'],  
            self.config['paths']['model']['save_dir'],  
            os.path.join(self.config['paths']['data']['results'],  
                        self.config['paths']['output']['feature_maps'])  
        ]  
        for directory in directories:  
            try:  
                os.makedirs(directory, exist_ok=True)  
                logging.debug(f"Created directory: {directory}")  
            except Exception as e:  
                logging.error(f"Error creating directory {directory}: {str(e)}")  
                raise  

    def validate_paths(self):  
        """验证关键路径是否存在"""  
        required_paths = {  
            'CLIP Model': self.config['paths']['model']['clip_model']['path'],  
            'Data Root': self.config['paths']['data']['root'],  
            'Model Save Dir': self.config['paths']['model']['save_dir']  
        }  
        
        for name, path in required_paths.items():  
            if path and not os.path.exists(path):  
                logging.warning(f"{name} path does not exist: {path}")  

    def get_model_path(self) -> str:
        """获取模型保存路径"""
        return os.path.join(
            self.config['paths']['model']['save_dir'],
            self.config['paths']['model']['best_model']
        )
    
    def __getitem__(self, key: str) -> Any:  
        """允许像字典一样访问配置"""  
        return self.config[key]  
    
    def get(self, key: str, default: Any = None) -> Any:  
        """安全获取配置值"""  
        return self.config.get(key, default)  
    
# 全局配置实例
_global_config = None

def get_config(config_path: str = None) -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config

def setup_logging(filename: str):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename)
        ]
    )

def setup_device():
    """设置设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device