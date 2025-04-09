import sys  
sys.path.append('/home/nw/Codes')  

import os  
import torch  
import torch.nn as nn  
import pandas as pd  
from torch.utils.data import DataLoader  
from Loaders.MLRSNet_loader import MLRSNetDataset  
from Models.CNN_MultiLabel_Classification.model_factory import create_model  

# 配置参数（与训练代码结构一致）  
class PredictConfig:  
    # 模型参数  
    MODEL_ARCH = 'resnet101'  # 必须与训练时一致  
    MODEL_PATH = '/home/nw/Codes/Jobs/CNN_MultiLabel_Classifier/model_save/best_model_epoch_50.pth'  
    NUM_CLASSES = 60  # 必须与训练时一致  
    
    # 设备参数（新增）  
    DEVICE_IDS  = [2, 3]
    MULTI_GPU = True if len(DEVICE_IDS) > 1 else False  

    # 数据参数  
    INPUT_DIR = '/home/Dataset/nw/Multilabel-Datasets/MLRSNet_dataset/Images/bridge'  # 支持目录或单文件路径  
    OUTPUT_CSV = os.path.join( INPUT_DIR, 'predictions.csv')
    
    # 推理参数  
    BATCH_SIZE = 192  # 与训练保持相同batch size  
    THRESHOLD = 0.5  

    @property  
    def main_device(self):  
        """根据设备ID列表确定主设备"""  
        if self.DEVICE_IDS:  
            return f'cuda:{self.DEVICE_IDS[0]}'  
        return 'cuda' if torch.cuda.is_available() else 'cpu' 

    

class Predictor:  
    def __init__(self, config):  
        self.config = config  
        self.model = self._load_model()  
        self.model.eval()  
        
    def _load_model(self):  
        """模型加载（与训练代码相同的设备处理逻辑）"""  
        model = create_model(  
            self.config.MODEL_ARCH,  
            self.config.NUM_CLASSES,  
            multi_gpu=self.config.MULTI_GPU,  
            device_ids=self.config.DEVICE_IDS
        )  

        # 加载权重（自动处理设备兼容性）  
        state_dict = torch.load(self.config.MODEL_PATH,   
                              map_location=self.config.device)  
        
        # 统一处理多GPU权重键名  
        if not isinstance(model, nn.DataParallel) and any(k.startswith('module.') for k in state_dict):  
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  
            
        model.load_state_dict(state_dict)  
        return model.to(self.config.main_device)  # 移动到主设备  
    
    def _create_dataset(self):  
        """创建预测数据集"""  
        if os.path.isfile(self.config.INPUT_DIR):  
            return [self.config.INPUT_DIR]  
            
        valid_ext = ('.png', '.jpg', '.jpeg')  
        return [  
            os.path.join(root, f)   
            for root, _, files in os.walk(self.config.INPUT_DIR)  
            for f in files   
            if f.lower().endswith(valid_ext)  
        ]  
    
    def predict(self):  
        """执行预测主流程"""  
        # 加载数据
        image_paths, _ = self._create_dataset()  
        dataset = MLRSNetDataset(  
            [(p, [0]*self.config.NUM_CLASSES) for p in image_paths],  # 假标签填充  
            self.model.preprocess  
        )  

        loader = DataLoader(  
            dataset,  
            batch_size=self.config.BATCH_SIZE,  
            num_workers=4 if self.config.MULTI_GPU else 2,  # 根据GPU数量调整  
            pin_memory=True  # 提升数据传输效率  
        )  
        
        # 执行批量预测  
        all_preds = []  
        with torch.no_grad():  
            for inputs, _ in loader:  
                inputs = inputs.to(self.config.main_device, non_blocking=True)  
                outputs = self.model(inputs)  
                probs = torch.sigmoid(outputs).cpu()  
                all_preds.extend((probs > self.config.THRESHOLD).int().tolist())  
        
        # 保存结果  
        pd.DataFrame({  
            'image_path': image_paths,  
            'predictions': all_preds  
        }).to_csv(self.config.OUTPUT_CSV, index=False)  
        
        print(f'[完成] 预测结果已保存至 {self.config.OUTPUT_CSV}')  

if __name__ == "__main__":  
    # 参数校验  
    config = PredictConfig()  
    
    assert os.path.exists(config.MODEL_PATH), f"模型文件不存在: {config.MODEL_PATH}"  
    assert os.path.exists(config.INPUT_DIR), f"输入路径不存在: {config.INPUT_DIR}"  
    if config.MULTI_GPU:  
        assert len(config.DEVICE_IDS) >= 2, "多GPU模式需要至少2个设备ID"  
    
    # 执行预测  
    predictor = Predictor(config)  
    predictor.predict()  