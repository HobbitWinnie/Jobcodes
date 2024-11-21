import os  
import logging  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import rasterio  
from torch.cuda.amp import autocast  
from tqdm import tqdm  
import numpy as np  

from config import get_config, setup_logging, setup_device  
from model import UNetWithCLIP  # 更改为你的模型类名  
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
        self.batch_size = config['predict']['batch_size']  
        self.tta = config['predict']['tta']  
        self.post_processing = config['predict']['post_processing']  
        
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
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  
        
        try:  
            # 加载图像数据  
            test_image, _, image_profile = load_and_save_data(  
                image_path=image_path,  
                label_path=None,  
                output_dir=None,  
                normalize=True,  
            )  
            
            # 获取nodata掩膜  
            with rasterio.open(image_path) as src:  
                nodata_value = src.nodata  
                original_data = src.read()  
                if nodata_value is not None:  
                    nodata_mask = (original_data[0] == nodata_value)  
                else:  
                    nodata_mask = np.zeros_like(original_data[0], dtype=bool)  

            # 分割图像为patches  
            patches = split_image_into_patches(  
                test_image,  
                self.patch_size,  
                self.overlap  
            )  
            
            # 批处理预测  
            predictions = []  
            self.model.eval()  
            
            with torch.no_grad():  
                for i in tqdm(range(0, len(patches), self.batch_size), desc="Processing patches"):  
                    batch_patches = patches[i:i + self.batch_size]  
                    batch_tensor = torch.tensor(batch_patches, dtype=torch.float32).to(self.device)  
                    
                    try:  
                        with autocast():  
                            if self.tta:  
                                pred = self._tta_predict(batch_tensor)  
                            else:  
                                output = self.model(batch_tensor)  
                                pred = F.softmax(output, dim=1)  
                            
                            pred = pred.cpu().numpy()  
                            predictions.extend([p for p in pred])  
                            
                    except RuntimeError as e:  
                        if "out of memory" in str(e):  
                            torch.cuda.empty_cache()  
                            logging.warning("GPU OOM, reducing batch size and retrying...")  
                            # 单个样本处理  
                            for single_patch in batch_patches:  
                                single_tensor = torch.tensor(single_patch, dtype=torch.float32).unsqueeze(0).to(self.device)  
                                with autocast():  
                                    output = self.model(single_tensor)  
                                    pred = F.softmax(output, dim=1)  
                                    pred = pred.cpu().numpy()  
                                    predictions.append(pred[0])  
                        else:  
                            logging.error(f"Error processing batch: {str(e)}")  
                            continue  
            
            if not predictions:  
                raise ValueError("No valid predictions generated")  

            # 重建完整图像  
            reconstructed_prediction = reconstruct_image_from_patches(  
                predictions,  
                (test_image.shape[1], test_image.shape[2]),  
                self.patch_size,  
                self.overlap  
            )  
            
            # # 后处理  
            # if self.post_processing['enable']:  
            #     reconstructed_prediction = self._apply_post_processing(reconstructed_prediction)  
            
            # 应用nodata掩膜  
            reconstructed_prediction[nodata_mask] = 0  

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
    
    def _tta_predict(self, x):  
        """测试时增强预测"""  
        # 原始预测  
        pred = self.model(x)  
        pred = F.softmax(pred, dim=1)  
        
        # 水平翻转  
        x_flip = torch.flip(x, [-1])  
        pred_flip = self.model(x_flip)  
        pred_flip = F.softmax(pred_flip, dim=1)  
        pred_flip = torch.flip(pred_flip, [-1])  
        
        # 垂直翻转  
        x_flip = torch.flip(x, [-2])  
        pred_flip_v = self.model(x_flip)  
        pred_flip_v = F.softmax(pred_flip_v, dim=1)  
        pred_flip_v = torch.flip(pred_flip_v, [-2])  
        
        # 平均预测结果  
        pred = (pred + pred_flip + pred_flip_v) / 3  
        return pred  
    
    def _apply_post_processing(self, prediction):  
        """应用后处理"""  
        from skimage import morphology  
        
        # 移除小区域  
        if self.post_processing.get('min_size', 0) > 0:  
            for i in range(1, prediction.max() + 1):  
                mask = prediction == i  
                processed = morphology.remove_small_objects(  
                    mask,   
                    min_size=self.post_processing['min_size']  
                )  
                prediction[mask != processed] = 0  
        
        # 平滑处理  
        if self.post_processing.get('smoothing', False):  
            for i in range(1, prediction.max() + 1):  
                mask = prediction == i  
                processed = morphology.binary_closing(mask)  
                processed = morphology.binary_opening(processed)  
                prediction[mask != processed] = 0  
                
        return prediction  

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
        # 加载CLIP模型  
        clip_model = config.load_clip_model(device)  
        logging.info("CLIP model loaded successfully")  
        
        # 初始化模型  
        model = UNetWithCLIP(clip_model=clip_model, **config['model'])  
        
        # 加载预训练权重  
        model_path = config.get_model_path()  
        logging.info(f"Loading model from {model_path}")  
        
        checkpoint = torch.load(model_path, map_location=device)  
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:  
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