import os
import yaml
from typing import Dict, Any
import logging
import torch

class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置
        Args:
            config_path: 配置文件路径,默认为None使用默认配置
        """
        self.config = self._get_default_config()
        
        if config_path is not None:
            self.load_config(config_path)
            
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
                    'best_model': 'best_model.pth'
                },
                'input': {
                    'train_image': 'GF2_train_image.tif',
                    'train_label': 'train_label.tif',
                    'train_mask': 'train_mask.tif',
                    'test_image': 'GF2_test_image.tif'
                },
                'output': {
                    'train_mask_result': 'train_mask_results.tif',
                    'test_image_result': 'GF2_test_image_results.tif'
                }
            },
            'dataset': {
                'patch_size': 256,
                'patch_number': 8000,
                'train_val_split': 0.8,
                'num_classes': 9
            },
            'model': {
                'in_channels': 4,
                'out_channels': 9,
                'initial_features': 128,
                'dropout_rate': 0.2
            },
            'training': {
                'epochs': 2000,
                'batch_size': 192,
                'learning_rate': 5e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.001,
                'scheduler_T0': 30,
                'scheduler_T_mult': 2,
                'patience': 100,
                'max_grad_norm': 1.0,
                'loss_weights': [0.5, 0.5],
                'ignore_index': 0
            },
            'predict': {
                'overlap': 64,
                'batch_size': 32
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """从文件加载配置"""
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            self.config = self._update_nested_dict(self.config, file_config)
    
    def save_config(self, config_path: str) -> None:
        """保存配置到文件"""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _update_nested_dict(self, d1: Dict, d2: Dict) -> Dict:
        """递归更新嵌套字典"""
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                self._update_nested_dict(d1[k], v)
            else:
                d1[k] = v
        return d1
    
    def get_model_path(self) -> str:
        """获取模型保存路径"""
        return os.path.join(
            self.config['paths']['model']['save_dir'],
            self.config['paths']['model']['best_model']
        )
    
    def create_directories(self) -> None:
        """创建必要的目录"""
        directories = [
            self.config['paths']['data']['process'],
            self.config['paths']['data']['results'],
            self.config['paths']['model']['save_dir']
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
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