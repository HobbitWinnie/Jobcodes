import os  
import logging  
import torch  
from typing import Dict, Any  

class Config:  
    """配置管理类"""  
    
    def __init__(self):  
        """初始化配置"""  
        self.config = {  
            'paths': {  
                'data': {  
                    'images': '/home/Dataset/nw/Segmentation/CpeosTest/images',  
                    'process': '/home/Dataset/nw/Segmentation/CpeosTest/image_process',  
                    'results': '/home/Dataset/nw/Segmentation/CpeosTest/result'  
                },  
                'model': {  
                    'save_dir': '/home/nw/Codes/RemoteCLIP/Image_segementation/model_save',  
                    'best_model': 'RemoteCLIP_UNet_best_model.pth',  
                    'clip_ckpt': '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-B-32.pt',  
                },  
                'input': {  
                    'train_image': 'GF2_train_image.tif',  
                    'train_label': 'train_label.tif',  
                    'test_image': 'GF2_test_image.tif'  
                }  
            },  
            'dataset': {  
                'patch_size': 224,  
                'patch_number': 3000,  
                'train_val_split': 0.8,  
                'num_classes': 9,  
                'num_workers': 4  
            },  
            'model': {  
                'in_channels': 4,  
                'initial_features': 128,  
                'dropout_rate': 0.2,  
                'model_name': 'ViT-B-32'
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
                'clip_grad_norm': 1.0,  
                'ignore_index': 0  
            }  
        }  
        
        self._validate_config()  
    
    def _validate_config(self):  
        """验证配置有效性"""  
        # 验证关键路径  
        required_paths = [  
            self.config['paths']['data']['images'],  
            self.config['paths']['model']['clip_ckpt']  
        ]  
        for path in required_paths:  
            if not os.path.exists(path):  
                logging.warning(f"Required path does not exist: {path}")  
    
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

# 全局配置实例  
_global_config = None  

def get_config() -> Config:  
    """获取全局配置实例"""  
    global _global_config  
    if _global_config is None:  
        _global_config = Config()  
    return _global_config  

def setup_logging(filename: str):  
    """设置日志"""  
    os.makedirs(os.path.dirname(filename), exist_ok=True)  
    logging.basicConfig(  
        level=logging.INFO,  
        format='%(asctime)s [%(levelname)s] %(message)s',  
        handlers=[  
            logging.StreamHandler(),  
            logging.FileHandler(filename)  
        ]  
    )  

def setup_device() -> torch.device:  
    """设置计算设备"""  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logging.info(f"Using device: {device}")  
    return device