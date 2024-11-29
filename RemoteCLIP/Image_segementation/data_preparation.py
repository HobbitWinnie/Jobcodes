
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path  
from config import get_config
import logging
from datetime import datetime
import rasterio
import tifffile  


# 设置日志级别和格式  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
) 

def load_and_save_data(image_path, label_path, output_dir, normalize = True):  
    # 如果指定了输出目录，则创建相应的目录结构  
    if output_dir is not None:  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        output_dir = os.path.join(output_dir, timestamp)  
        os.makedirs(output_dir, exist_ok=True)  
    
    # 加载并处理图像数据  
    try:  
        logging.info(f"Loading image from {image_path}")  
        with rasterio.open(image_path) as src:  
            image = src.read()  # 图像形状为 (bands, height, width)  
            image_nodata = int(src.nodata)  
            image_meta = src.meta  
            
        # **只保留前三个波段**  
        if image.shape[0] >= 3:  
            image = image[:3, :, :]  
        else:  
            raise ValueError("图像的波段数少于 3，无法提取前三个波段")  

        # 处理无效值  
        image_mask = np.ones_like(image[0], dtype=bool)  
        if image_nodata is not None:  
            image_mask = (image[0] != image_nodata)  
            
        # 数据有效性检查  
        if np.all(~image_mask):  
            raise ValueError("No valid data in image")  
            
        # # 替换无效值为0  
        # image = np.where(np.broadcast_to(image_mask, image.shape), image, 0)  
                
        # **像素值缩放到 0-255 范围**  
        image_processed = np.zeros_like(image, dtype=image.dtype)  
        for i in range(image.shape[0]):  
            valid_data = image[i][image_mask]  
            if len(valid_data) > 0:  
                # 计算第 1 和 99 百分位数  
                min_val = np.percentile(valid_data, 0.5)  
                max_val = np.percentile(valid_data, 99.5)  
                # 裁剪图像数据到指定范围  
                image_clipped = np.clip(image[i], min_val, max_val)  

                # 缩放到 0-1  
                image_scaled = (image_clipped - min_val) / (max_val - min_val)  
                image_processed[i] = image_scaled  
            else:  
                # 如果没有有效数据，直接复制原始数据  
                image_processed[i] = image[i]  
       
        # 将图像转换为 float32 类型，并确保值在 0-1 范围内  
        image = image_processed.astype(np.float32)  # 0-255时，数据类型为uint8 

        for i in range(image.shape[0]):  
            print(f"波段 {i+1} 的数据范围：{image[i].min()} - {image[i].max()}")
            
        # 如果指定了输出目录，则保存处理后的图像  
        if output_dir is not None:  
            processed_image_path = os.path.join(output_dir, 'processed_image.tif')  
            with rasterio.open(  
                processed_image_path,  
                'w',  
                driver='GTiff',  
                height=image.shape[1],  
                width=image.shape[2],  
                count=image.shape[0],  
                dtype=image.dtype,  
                crs=image_meta['crs'],  
                transform=image_meta['transform']  
            ) as dst:  
                dst.write(image)  
                if normalize:  
                    dst.write_mask(image_mask.astype(np.uint8))  
        
    except Exception as e:  
        logging.error(f"Error processing image: {e}")  
        raise  
    
    # 加载并处理标签数据  
    labels = None  
    if label_path:  
        try:  
            logging.info(f"Loading labels from {label_path}")  
            with rasterio.open(label_path) as src:  
                labels = src.read(1)  
                labels_nodata = int(src.nodata)  
                
            # 处理标签无效值  
            label_mask = np.ones_like(labels, dtype=bool)  
            if labels_nodata is not None:  
                label_mask = (labels != labels_nodata)  
                
            # 替换无效值为0  
            labels = np.where(label_mask, labels, 0)  
            
            # 如果指定了输出目录，则保存处理后的标签  
            if output_dir is not None:  
                processed_label_path = os.path.join(output_dir, 'processed_labels.tif')  
                with rasterio.open(  
                    processed_label_path,  
                    'w',  
                    driver='GTiff',  
                    height=labels.shape[0],  
                    width=labels.shape[1],  
                    count=1,  
                    dtype=labels.dtype,  
                    crs=image_meta['crs'],  
                    transform=image_meta['transform']  
                ) as dst:  
                    dst.write(labels, 1)  
                    dst.write_mask(label_mask.astype(np.uint8))  
                
        except Exception as e:  
            logging.error(f"Error processing labels: {e}")  
            raise  
    
    # 记录处理信息  
    logging.info(f"Image shape: {image.shape}, dtype: {image.dtype}")  
    if labels is not None:  
        logging.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")  
    
    if output_dir is not None:  
        logging.info(f"Data processing completed. Results saved to {output_dir}")  
        
    return image, labels, image_meta, image_nodata


def preprocess_and_save_patches(image, labels, patch_size, num_patches, save_dir, image_nodata_value):  
    """  
    从处理过的图像和标签中随机采样指定数量的图像块，并保存到指定目录。  

    Args:  
        image: 处理后的图像数据，形状为 (C, H, W)  
        labels: 标签数据，形状为 (H, W)  
        patch_size: 图像块的大小  
        num_patches: 采样的图像块数量  
        save_dir: 保存图像块的目录  
    """  
    os.makedirs(save_dir, exist_ok=True)  
    images_dir = os.path.join(save_dir, 'images')  
    labels_dir = os.path.join(save_dir, 'labels')  
    os.makedirs(images_dir, exist_ok=True)  
    os.makedirs(labels_dir, exist_ok=True)  
    
    h, w = image.shape[1], image.shape[2]  
    saved_patches = 0  # 已经保存的图像块数量  
    attempts = 0  # 采样尝试次数  
    
    max_attempts = num_patches * 30 # 最大尝试次数，防止无限循环  
    
    while saved_patches < num_patches and attempts < max_attempts:  
        attempts += 1  
    
        x = random.randint(0, w - patch_size)  
        y = random.randint(0, h - patch_size)  
    
        # 提取图像块和标签块  
        image_patch = image[:, y:y+patch_size, x:x+patch_size]  # (C, H, W)  
        label_patch = labels[y:y+patch_size, x:x+patch_size]  
    
        # 检查标签块中零值的比例  
        zero_ratio = np.sum(label_patch == 0) / (patch_size * patch_size)  
        if zero_ratio > 0.2:  
            continue  # 舍弃该图像块，继续下一个采样  
    
        # 忽略背景值
        if np.any(image_patch == image_nodata_value):  
            continue  # 跳过包含无效值的图像块  
    
        # 转换图像格式 (C, H, W) -> (H, W, C)  
        image_patch_transposed = np.transpose(image_patch, (1, 2, 0))  
    
        # 保存图像和标签  
        image_filename = os.path.join(images_dir, f"image_patch_{saved_patches}.tif")  
        label_filename = os.path.join(labels_dir, f"label_patch_{saved_patches}.tif")  
    
        # 使用 tifffile 保存，保留原始数据类型和精度  
        tifffile.imwrite(image_filename, image_patch_transposed)  
        tifffile.imwrite(label_filename, label_patch)  
    
        saved_patches += 1  # 成功保存，计数加1  
    
    if attempts >= max_attempts:  
        print(f"在最大尝试次数 {max_attempts} 次后，未能采样足够的图像块, 成功保存了 {saved_patches} 个图像块。")  
    else:  
        print(f"成功保存了 {saved_patches} 个图像块。")


if __name__ == '__main__':

    # 加载原始数据  
    print("开始加载数据...")  
    config = get_config()

    image_path = Path(config['paths']['data']['images']) / config['paths']['input']['train_image']  
    label_path = Path(config['paths']['data']['images']) / config['paths']['input']['train_label']  

    image, labels, _, image_nodata_value = load_and_save_data(  
        image_path=image_path,  
        label_path=label_path,  
        output_dir=config['paths']['data']['process']  
    )  

    # 设置参数并运行预处理  
    save_dir = '/home/Dataset/nw/Segmentation/CpeosTest/train_0_1'  
    patch_size=config['dataset']['patch_size']
    num_patches=config['dataset']['patch_number']

    preprocess_and_save_patches(image, labels, patch_size, num_patches, save_dir, image_nodata_value)