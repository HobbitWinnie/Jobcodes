import numpy as np
import rasterio
import logging
import os
from datetime import datetime
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt

def load_and_save_data(
    image_path: str,
    label_path: Optional[str] = None,
    output_dir: str = './processed_data',
    normalize: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    加载并处理遥感图像和标签数据，支持数据预处理和保存功能
    
    Args:
        image_path: 图像文件路径
        label_path: 标签文件路径（可选）
        output_dir: 输出目录
        normalize: 是否进行归一化
        save_plots: 是否保存可视化图
        
    Returns:
        处理后的图像数据、标签数据和元数据
    """
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载并处理图像数据
    try:
        logging.info(f"Loading image from {image_path}")
        with rasterio.open(image_path) as src:
            image = src.read()
            image_nodata = int(src.nodata)
            image_meta = src.meta
            
        # 处理无效值
        image_mask = np.ones_like(image[0], dtype=bool)
        if image_nodata is not None:
            image_mask = (image[0] != image_nodata)
            
        # 数据有效性检查
        if np.all(~image_mask):
            raise ValueError("No valid data in image")
            
        # 替换无效值为0
        image = np.where(np.broadcast_to(image_mask, image.shape), image, 0)
        
        # 数据归一化
        if normalize:
            image_normalized = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[0]):
                valid_data = image[i][image_mask]
                if len(valid_data) > 0:
                    min_val = np.percentile(valid_data, 1)
                    max_val = np.percentile(valid_data, 99)
                    image_normalized[i] = np.clip((image[i] - min_val) / (max_val - min_val + 1e-8), 0, 1)
            image = image_normalized
            
        # 保存处理后的图像
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
            
            # 保存处理后的标签
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
           
    logging.info(f"Data processing completed. Results saved to {output_dir}")
    logging.info(f"Image shape: {image.shape}, dtype: {image.dtype}")
    if labels is not None:
        logging.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
        
    return image, labels, image_meta

# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 示例路径
    image_path = "/home/Dataset/nw/Segmentation/CpeosTest/images/GF2_train_image.tif"
    label_path = "/home/Dataset/nw/Segmentation/CpeosTest/images/train_label.tif"
    output_dir = "/home/Dataset/nw/Segmentation/CpeosTest/image_process"
    
    try:
        image, labels, meta = load_and_save_data(
            image_path=image_path,
            label_path=label_path,
            output_dir=output_dir,
            normalize=True,
        )
        logging.info("Data loading and processing successful")
    except Exception as e:
        logging.error(f"Error in data processing: {e}")