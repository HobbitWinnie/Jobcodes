import os
import torch
import rasterio
import logging
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast

from utils import load_and_save_data
from model import UNet
from dataset import split_image_into_patches, reconstruct_image_from_patches

def predict_image(model, image_path, output_path, config, device):
    """预测单张图像"""
    logging.info(f"Predicting for {image_path}")
    
    # 加载图像     
    test_image, _, image_profile = load_and_save_data(
        image_path=image_path,
        label_path=None,
        output_dir=None,
        normalize=True,
    )
    
    # 分割图像
    patches = split_image_into_patches(
        test_image, 
        config['patch_size'], 
        config['overlap']
    )
    
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for patch in tqdm(patches, desc="Processing patches"):
            patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
            
            with autocast():
                output = model(patch)
                pred = F.softmax(output, dim=1).squeeze().cpu().numpy()
            predictions.append(pred)
    
    # 重建完整图像
    reconstructed_prediction = reconstruct_image_from_patches(
        predictions,
        test_image.shape,
        config['patch_size'],
        config['overlap']
    )
    
    # 保存预测结果
    image_profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(output_path, 'w', **image_profile) as dst:
        dst.write(reconstructed_prediction.astype(rasterio.uint8), 1)
    
    logging.info(f"Prediction saved to {output_path}")

def main():
    # 配置
    config = {
        'paths': {
            'model_path': '/home/nw/Codes/Segement_Models/model_save/best_model.pth',
            'image_root': '/home/Dataset/nw/Segmentation/CpeosTest/images',
            'result_dir': '/home/Dataset/nw/Segmentation/CpeosTest/result'
        },
        'predict': {
            'patch_size': 256,
            'overlap': 64
        },
        'model': {
            'in_channels': 4,
            'out_channels': 8,
            'initial_features': 64,
            'dropout_rate': 0.2
        }
    }
    
    # 创建输出目录
    os.makedirs(config['paths']['result_dir'], exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    try:
        checkpoint = torch.load(config['paths']['model_path'], map_location=device)
        model = UNet(**config['model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    
    # 设置输入输出路径
    test_img_paths = [
        os.path.join(config['paths']['image_root'], 'train_mask.tif'),
        os.path.join(config['paths']['image_root'], 'GF2_test_image.tif')
    ]
    output_paths = [
        os.path.join(config['paths']['result_dir'], 'train_mask_gptUnet_results.tif'),
        os.path.join(config['paths']['result_dir'], 'GF2_test_image_gptUnet_results.tif')
    ]
    
    # 预测每张图像
    for img_path, out_path in zip(test_img_paths, output_paths):
        try:
            predict_image(
                model=model,
                image_path=img_path,
                output_path=out_path,
                config=config['predict'],
                device=device
            )
        except Exception as e:
            logging.error(f"Error predicting {img_path}: {e}")
            continue

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    main()