
import logging
import numpy as np
import rasterio
import torch
import torch.nn as nn
from typing import Tuple
from data_loader import GeoTIFFLoader
from config import PredictConfig
from Models.Pixel_based_CNN_classification.ResNet50 import ResNet50


logger = logging.getLogger(__name__)

class ImagePredictor:
    """遥感影像预测器"""
    
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()
        self.device = torch.device(config.device)
        
    def _load_model(self) -> nn.Module:
        """加载训练好的模型"""
        try:
            model = ResNet50(num_classes=self.config.num_classes)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.load_state_dict(torch.load(self.config.model_path))
            return model.to(self.device).eval()
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """图像预处理"""
        pad_size = self.config.patch_size // 2
        padded_image = np.pad(
            image,
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode='edge'
        )
        return padded_image

    def _batch_predict(self, patches: torch.Tensor) -> np.ndarray:
        """批量预测"""
        with torch.no_grad():
            outputs = self.model(patches)
            return outputs.argmax(dim=1).cpu().numpy()

    def predict(self) -> None:
        """执行完整预测流程"""
        # 加载影像数据
        image, meta, _ = GeoTIFFLoader.load_geotiff(self.config.input_image)
        original_shape = image.shape[1:]  # 获取原始影像H,W
        
        # 预处理
        padded_image = self._preprocess_image(image)
        result = np.full(original_shape, self.config.nodata_value, dtype=np.float32)
        
        # 逐像素预测
        for row in range(original_shape[0]):
            for col in range(original_shape[1]):
                patch = padded_image[
                    :, 
                    row:row+self.config.patch_size,
                    col:col+self.config.patch_size
                ]
                patch_tensor = torch.tensor(patch, dtype=torch.float32)
                patch_tensor = patch_tensor.unsqueeze(0).to(self.device)
                
                pred = self._batch_predict(patch_tensor)[0]
                result[row, col] = pred
                
        
        # 保存结果
        self._save_result(result, meta)

    def _save_result(self, result: np.ndarray, meta: dict) -> None:
        """保存预测结果"""
        meta.update({
            'dtype': rasterio.float32,
            'count': 1,
            'nodata': self.config.nodata_value
        })
        
        try:
            with rasterio.open(self.config.output_path, 'w', **meta) as dst:
                dst.write(result, 1)
            logger.info(f"预测结果已保存至 {self.config.output_path}")
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            raise


def main():  
    # 配置日志  
    logging.basicConfig(  
        level=logging.INFO,  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  
    )  

    # 初始化配置  
    config = PredictConfig(  
        patch_size=11,  
        nodata_value=15.0  
    )  
    
    # 创建预测器并执行预测  
    predictor = ImagePredictor(config)  
    predictor.predict()  

if __name__ == "__main__":  
    main()  