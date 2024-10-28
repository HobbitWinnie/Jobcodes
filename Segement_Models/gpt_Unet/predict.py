import os  
import torch  
import rasterio  
import logging  
import numpy as np  
import torch.nn as nn  
import torch.nn.functional as F  
from tqdm import tqdm  
from torch.cuda.amp import autocast  

from config import get_config, setup_logging, setup_device  
from utils import load_and_save_data  
from model import UNet  
from dataset import split_image_into_patches, reconstruct_image_from_patches  

def predict_image(model, image_path, output_path, config, device):  
    """预测单张图像"""  
    logging.info(f"Predicting for {image_path}")  
    
    try:  
        # 加载图像数据  
        test_image, _, image_profile = load_and_save_data(  
            image_path=image_path,  
            label_path=None,  
            output_dir=None,  
            normalize=True,  
        )  
        logging.info(f"Image loaded successfully, shape: {test_image.shape}")  
        
        # 分割图像为patches  
        patches, indices = split_image_into_patches(  
            test_image,   
            config['dataset']['patch_size'],  
            config['predict']['overlap']  
        )  
        logging.info(f"Split into {len(patches)} patches")  
        
        # 预测  
        predictions = []  
        model.eval()  
        
        with torch.no_grad():  
            for patch in tqdm(patches, desc="Processing patches"):  
                patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  
                
                try:  
                    with autocast():  
                        output = model(patch)  
                        pred = F.softmax(output, dim=1)  
                        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()  
                    predictions.append(pred)  
                except Exception as e:  
                    logging.error(f"Error processing patch: {str(e)}")  
                    continue  
        
        # 重建完整图像  
        reconstructed_prediction = reconstruct_image_from_patches(  
            predictions=predictions,  
            indices=indices,  
            original_shape=test_image.shape,  
            patch_size=config['dataset']['patch_size'],  
            overlap=config['predict']['overlap']  
        )  
        
        # 保存预测结果  
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  
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
    """加载模型"""  
    try:  
        model_path = config.get_model_path()  
        logging.info(f"Loading model from {model_path}")  
        
        checkpoint = torch.load(model_path, map_location=device)  
        model = UNet(**config['model'])  
        
        if 'model_state_dict' in checkpoint:  
            state_dict = checkpoint['model_state_dict']  
        else:  
            state_dict = checkpoint  
            
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
    try:  
        # 获取配置  
        config = get_config()  
        
        # 创建必要的目录  
        config.create_directories()  
        
        # 设置日志  
        log_path = os.path.join(config['paths']['data']['results'], 'predict.log')  
        setup_logging(log_path)  
        
        # 设置设备  
        device = setup_device()  
        
        # 加载模型  
        model = load_model(config, device)  
        
        # 构建输入输出路径  
        test_img_paths = [  
            os.path.join(config['paths']['data']['images'],   
                        config['paths']['input']['train_mask']),  
            os.path.join(config['paths']['data']['images'],   
                        config['paths']['input']['test_image'])  
        ]  
        
        output_paths = [  
            os.path.join(config['paths']['data']['results'],   
                        config['paths']['output']['train_mask_result']),  
            os.path.join(config['paths']['data']['results'],   
                        config['paths']['output']['test_image_result'])  
        ]  
        
        # 预测每张图像  
        for img_path, out_path in zip(test_img_paths, output_paths):  
            if not os.path.exists(img_path):  
                logging.error(f"Image file not found: {img_path}")  
                continue  
                
            try:  
                predict_image(  
                    model=model,  
                    image_path=img_path,  
                    output_path=out_path,  
                    config=config,  
                    device=device  
                )  
            except Exception as e:  
                logging.error(f"Error predicting {img_path}: {str(e)}")  
                continue  
                
        logging.info("Prediction completed successfully")  
        
    except Exception as e:  
        logging.error(f"Error in main pipeline: {str(e)}")  
        raise  

if __name__ == "__main__":  
    main()