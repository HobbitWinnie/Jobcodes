import os  
import logging  
import torch  


class Config:  
    """配置管理类"""  
    
    def __init__(self):  
        """初始化配置"""  
        self.config = {  
            'paths': {  
                'data': {  
                    'images': '/home/Dataset/nw/Segmentation/CpeosTest/images',  
                    'process': '/home/Dataset/nw/Segmentation/CpeosTest/image_process',  
                    'results': '/home/Dataset/nw/Segmentation/CpeosTest/result',
                    'image_dir': '/home/Dataset/nw/Segmentation/CpeosTest/train_4channel/images',
                    'label_dir': '/home/Dataset/nw/Segmentation/CpeosTest/train_4channel/labels'
                },  
                'model': {  
                    'save_dir': '/home/nw/Codes/RemoteCLIP/Image_segementation/model_save',  
                    'best_model': '/home/nw/Codes/RemoteCLIP/Image_segementation/model_save/20241203_101655/best_model.pth',  
                    'clip_ckpt': '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-RN50.pt',  
                },  
                'input': {  
                    'train_image': 'GF2_train_image.tif',  
                    'train_label': 'train_label.tif',  
                    'train_mask': 'train_mask.tif',
                    'test_image': 'GF2_test_image.tif'  
                } ,
                'output': {
                    'train_mask_result': 'train_mask_results_UNetWithCLIP.tif',
                    'test_image_result': 'GF2_test_image_results_UNetWithCLIP.tif'
                }
            },  
            'dataset': {  
                'patch_size': 224,  
                'patch_number': 5000,  
                'train_val_split': 0.8,  
                'num_classes': 9,   
                'num_workers': 4  
            },  
            'model': {  
                'in_channels': 4,  
                'initial_features': 128,  
                'dropout_rate': 0.2,  
                'model_name': 'RN50'
            },  
            'training': {  
                'epochs': 20000,  
                'batch_size': 128,  
                'learning_rate': 1e-4,
                'min_lr': 1e-7,  
                'weight_decay': 0.001,  
                'patience': 100,  
                'clip_grad_norm': 1.0,  
                'ignore_index': 0,
                'val_frequency': 10,
                'max_grad_norm': 0.5,  # 梯度裁剪阈值  
                'scheduler_T0': 200,    # CosineAnnealingWarmRestarts的初始周期  
                'scheduler_T_mult': 2, # 周期倍增因子 
                'loss_weights': [0.3, 0.7],  # CE损失和Dice损失的权重  
                'epsilon': 1e-7,         # 平滑参数  
            },
            'predict': {
                'overlap': 64,
                'batch_size': 32
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
    
    def create_directories(self):  
        """创建必要的目录"""  
        directories = [  
            self.config['paths']['data']['process'],  
            self.config['paths']['data']['results'],  
            self.config['paths']['model']['save_dir']  
        ]  
        for directory in directories:  
            os.makedirs(directory, exist_ok=True)  
    
    def __getitem__(self, key: str):  
        """允许像字典一样访问配置"""  
        return self.config[key]  

# 全局配置实例  
_global_config = None  

def get_config():  
    """获取全局配置实例"""  
    global _global_config  
    if _global_config is None:  
        _global_config = Config()  
    return _global_config  

def setup_device():  
    """设置计算设备"""  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logging.info(f"Using device: {device}")  
    return device