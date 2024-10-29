import os  
import logging  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import rasterio  
from torch.cuda.amp import autocast  
from tqdm import tqdm  

from config import get_config, setup_logging, setup_device  
from model import UNet  
from utils import load_and_save_data  
from dataset import split_image_into_patches, reconstruct_image_from_patches  

class Predictor:  
    """预测类"""  
    
    def __init__(self, model, config, device):  
        """  
        初始化预测器  
        
        Args:  
            model: 训练好的模型  
            config: Config实例  
            device: 计算设备  
        """  
        self.model = model  
        self.config = config  
        self.device = device  
        self.patch_size = config['dataset']['patch_size']  
        self.overlap = config['predict']['overlap']  
        
    def predict_images(self, image_paths, output_paths):  
        """  
        预测多张图像  
        
        Args:  
            image_paths: 输入图像路径列表  
            output_paths: 输出预测结果路径列表  
        """  
        for img_path, out_path in zip(image_paths, output_paths):  
            if not os.path.exists(img_path):  
                logging.error(f"Image file not found: {img_path}")  
                continue  
                
            try:  
                self.predict_single_image(img_path, out_path)  
            except Exception as e:  
                logging.error(f"Error predicting {img_path}: {str(e)}")  
    
    def predict_single_image(self, image_path, output_path):  
        """  
        预测单张图像  
        
        Args:  
            image_path: 输入图像路径  
            output_path: 输出预测结果路径  
        """  
        logging.info(f"Predicting for {image_path}")  
        
        # 确保输出目录存在  
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  
        
        try:  
            # 加载图像数据  
            test_image, _, image_profile = load_and_save_data(  
                image_path=image_path,  
                label_path=None,  
                output_dir=None,  
                normalize=True,  
            )  
            
            # 分割图像为patches  
            patches = split_image_into_patches(  
                test_image,  
                self.patch_size,  
                self.overlap  
            )  
            
            # 预测  
            predictions = []  
            self.model.eval()  
            
            # 使用tqdm显示进度  
            with torch.no_grad():  
                for patch in tqdm(patches, desc="Processing patches"):  
                    patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(self.device)  
                    
                    try:  
                        with autocast():  
                            output = self.model(patch)  
                            pred = F.softmax(output, dim=1)  
                            pred = pred.squeeze(0).cpu().numpy()  # 只移除batch维度  
                        predictions.append(pred)  
                    except Exception as e:  
                        logging.error(f"Error processing patch: {str(e)}")  
                        continue  
            
            # 打印shape信息用于调试  
            if predictions:  
                logging.debug(f"Prediction shape: {predictions[0].shape}")  

            # 重建完整图像  
            reconstructed_prediction = reconstruct_image_from_patches(  
                predictions,  
                (test_image.shape[1], test_image.shape[2]),  # 只传入高度和宽度  
                self.patch_size,  
                self.overlap  
            )  
            
            # 保存预测结果  
            image_profile.update(  
                dtype=rasterio.uint8,  
                count=1,  
                nodata=0,  
                compress='lzw'  
            )  
            with rasterio.open(output_path, 'w', **image_profile) as dst:  
                dst.write(reconstructed_prediction.astype(rasterio.uint8), 1)  
            logging.info(f"Prediction saved to {output_path}")  
            
        except Exception as e:  
            logging.error(f"Error in prediction pipeline: {str(e)}")  
            raise  

def load_model(config, device):  
    """  
    加载模型  
    
    Args:  
        config: Config实例  
        device: 计算设备  
        
    Returns:  
        加载好的模型  
    """  
    try:  
        model_path = config.get_model_path()  
        logging.info(f"Loading model from {model_path}")  
        
        checkpoint = torch.load(model_path, map_location=device)  
        model = UNet(**config['model'])  
        
        if 'model_state_dict' in checkpoint:  
            state_dict = checkpoint['model_state_dict']  
        else:  
            state_dict = checkpoint  
        
        # 处理DataParallel的状态字典  
        if list(state_dict.keys())[0].startswith('module.'):  
            state_dict = {k[7:]: v for k, v in state_dict.items()}  
        
        model.load_state_dict(state_dict)  
        
        if torch.cuda.device_count() > 1:  
            model = nn.DataParallel(model)  
        
        model.to(device)  
        model.eval()  
        
        logging.info("Model loaded successfully")  
        return model  
        
    except Exception as e:  
        logging.error(f"Error loading model: {str(e)}")  
        raise  

def main():  
    """主函数"""  
    try:  
        # 获取配置并初始化  
        config = get_config()  
        config.create_directories()  
        
        # 设置日志  
        log_path = os.path.join(config['paths']['data']['results'], 'predict.log')  
        setup_logging(log_path)  
        
        # 设置设备  
        device = setup_device()  
        
        # 加载模型  
        model = load_model(config, device)  
        
        # 构建输入输出路径  
        image_paths = [  
            os.path.join(  
                config['paths']['data']['images'],  
                config['paths']['input']['train_mask']  
            ),  
            os.path.join(  
                config['paths']['data']['images'],  
                config['paths']['input']['test_image']  
            )  
        ]  
        
        output_paths = [  
            os.path.join(  
                config['paths']['data']['results'],  
                config['paths']['output']['train_mask_result']  
            ),  
            os.path.join(  
                config['paths']['data']['results'],  
                config['paths']['output']['test_image_result']  
            )  
        ]  
        
        # 创建预测器并执行预测  
        predictor = Predictor(model, config, device)  
        predictor.predict_images(image_paths, output_paths)  
        
        logging.info("Prediction completed successfully")  
        
    except Exception as e:  
        logging.error(f"Error in prediction pipeline: {str(e)}")  
        raise  

if __name__ == "__main__":  
    main()