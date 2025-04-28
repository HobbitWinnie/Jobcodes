import os  
import logging  
from typing import Dict, Any, Optional, Union
import json  
from pathlib import Path  


class ConfigException(Exception):  
    pass  

class Config:  
    """  
    全局配置  
    - 支持点式、字典式访问  
    - 支持保存/加载json  
    - 支持目录校验、自动创建  
    """  
    def __init__(self, from_file: Optional[str] = None):  
        if from_file:  
            self.config = self._load_json(from_file)  
        else:  
            self.config: Dict[str, Any] = self._default_config()  
        self._validate_config()  
        self.create_directories()  
   
    @staticmethod  
    def _default_config(self) -> Dict[str, Any]:  
        # 此处可以加更多默认字段  
        base_dir = Path("/home/Dataset/nw/Segmentation/CpeosTest")  
        return {  
            'paths': {  
                'data': {  
                    'images': base_dir / "images",  
                    'process': base_dir / "image_process",  
                    'results': base_dir / "result",  
                    'image_dir': base_dir / "train_4channel" / "images",  
                    'label_dir': base_dir / "train_4channel" / "labels"  
                },  
                'model': {  
                    'save_dir': '/home/nw/Codes/Jobs/RemoteCLIP_based_Jobs/image_segmentation',  
                    'clip_ckpt': '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-RN50.pt',  
                },  
                'input': {  
                    'train_image': 'GF2_train_image.tif',  
                    'train_label': 'train_label.tif',  
                    'train_mask': 'train_mask.tif',  
                    'test_image': 'GF2_test_image.tif'  
                },  
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
                'model_type':'UNetWithReCLIPResNet',
                'model_name': 'RN50',
                'in_channels': 4,  
                'initial_features': 128,  
                'dropout_rate': 0.2,  
            },  
            'training': {  
                'epochs': 20000,  
                'batch_size': 96,  
                'learning_rate': 1e-5,  
                'min_lr': 1e-6,  
                'weight_decay': 0.001,  
                'patience': 100,  
                'clip_grad_norm': 1.0,  
                'ignore_index': 0,  
                'val_frequency': 10,  
                'max_grad_norm': 0.5,  
                'scheduler_T0': 200,  
                'scheduler_T_mult': 2,  
                'loss_weights': [0.3, 0.7],  
                'epsilon': 1e-7,  
            },  
            'predict': {  
                'overlap': 64,  
                'batch_size': 32  
            }  
        }  
    
    def _load_json(self, file_path: str) -> Dict[str, Any]:  
        if not os.path.isfile(file_path):  
            raise ConfigException(f"配置文件未找到: {file_path}")  
        with open(file_path, 'r') as f:  
            return json.load(f)  
        
    def save(self, file_path: str) -> None:  
        """保存配置为json"""  
        with open(file_path, 'w') as f:  
            json.dump(self.config, f, indent=4)  

    def to_dict(self) -> Dict[str, Any]:  
        """获取完整字典"""  
        return self.config  

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
        dirs = [  
            self.config['paths']['data']['process'],  
            self.config['paths']['data']['results'],  
            self.config['paths']['model']['save_dir']  
        ]  
        for d in dirs:  
            os.makedirs(d, exist_ok=True)  
    

    # 支持obj.attr 和 obj['key']  
    def __getitem__(self, key: str) -> Any:  
        """允许像字典一样访问配置"""  
        return self.config[key]  
    def __getattr__(self, key: str) -> Any:  
        if key in self.config:  
            return self.config[key]  
        raise AttributeError(f"'Config'未找到字段: {key}")  
    
    def update(self, new_config: Dict[str, Any]) -> None:  
        """支持字典批量覆盖更新参数"""  
        import collections.abc  
        def merge(d, u):  
            for k, v in u.items():  
                if isinstance(v, collections.abc.Mapping):  
                    d[k] = merge(d.get(k, {}), v)  
                else:  
                    d[k] = v  
            return d  
        merge(self.config, new_config)  


# 单例模式  
_config_instance: Optional[Config] = None  
def get_config(from_file: Optional[str] = None) -> Config:  
    global _config_instance  
    if _config_instance is None:  
        _config_instance = Config(from_file=from_file)  
    return _config_instance 