import sys  
sys.path.append('/home/nw/Codes')  

import os  
import torch  
import pandas as pd  
from PIL import Image  
from torch.utils.data import DataLoader  
from Loaders.MLRSNet_loader import MLRSNetDataset  
from Models.CNN_MultiLabel_Classifier.model_factory import create_model  

# 配置参数（与训练代码结构一致）  
class PredictConfig:  
    # 模型参数  
    MODEL_ARCH = 'resnet101'  # 必须与训练时一致  
    MODEL_PATH = '/home/nw/Codes/Jobs/CNN_MultiLabel_Classifier/model_save/best_model_epoch_50.pth'  
    NUM_CLASSES = 60  # 必须与训练时一致  
    
    # 数据参数  
    INPUT_DIR = '/home/Dataset/nw/Multilabel-Test'  # 支持目录或单文件路径  
    OUTPUT_CSV = './predictions.csv'  
    
    # 推理参数  
    BATCH_SIZE = 192  # 与训练保持相同batch size  
    THRESHOLD = 0.5  
    
    @property  
    def device(self):  
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
            device=self.config.device,  
            multi_gpu=False  
        )  
        state_dict = torch.load(self.config.MODEL_PATH,   
                              map_location=self.config.device)  
        
        # 处理多卡训练保存的权重  
        if all(k.startswith('module.') for k in state_dict):  
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  
        
        model.load_state_dict(state_dict)  
        return model.to(self.config.device)  
    
    def _create_dataset(self):  
        """创建预测数据集"""  
        if os.path.isfile(self.config.INPUT_DIR):  
            return [self.config.INPUT_DIR], []  
        
        image_paths = []  
        for root, _, files in os.walk(self.config.INPUT_DIR):  
            for file in files:  
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  
                    image_paths.append(os.path.join(root, file))  
        return image_paths, []  
    
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
            num_workers=4,  
            shuffle=False  
        )  
        
        # 执行批量预测  
        all_preds = []  
        with torch.no_grad():  
            for inputs, _ in loader:  
                inputs = inputs.to(self.config.device)  
                outputs = self.model(inputs)  
                probs = torch.sigmoid(outputs).cpu()  
                all_preds.extend((probs > self.config.THRESHOLD).int().tolist())  
        
        # 保存结果（与训练日志格式一致）  
        results = pd.DataFrame({ 
            'image_path': image_paths,  
            'predictions': all_preds  
        })  
        results.to_csv(self.config.OUTPUT_CSV, index=False)  
        print(f'[完成] 预测结果已保存至 {self.config.OUTPUT_CSV}')  

if __name__ == "__main__":  
    # 参数校验  
    config = PredictConfig()  
    assert os.path.exists(config.MODEL_PATH), f"模型文件不存在: {config.MODEL_PATH}"  
    assert os.path.exists(config.INPUT_DIR), f"输入路径不存在: {config.INPUT_DIR}"  
    
    # 执行预测  
    predictor = Predictor(config)  
    predictor.predict()  