import os  
import random  
import numpy as np  
from pathlib import Path  
import logging  
from datetime import datetime  
import rasterio  
import tifffile  
from typing import Tuple, Optional  
from config import get_config  
from utils.set_logging import setup_logging

  
def load_and_save_data(  
    image_path: str,  
    label_path: Optional[str],  
    output_dir: Optional[str],  
    normalize: bool = True  
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:  
    """  
    加载遥感影像与标签并处理；可保存中间结果。  
    """  
    if output_dir is not None:  
        output_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))  
        os.makedirs(output_dir, exist_ok=True)  
        logging.info(f"输出目录: {output_dir}")  
  
    # 图像读取和预处理  
    with rasterio.open(image_path) as src:  
        image = src.read()  # 图像形状为 (bands, height, width)  
        meta = src.meta.copy()  
        image_nodata = src.nodata

    mask_valid = np.ones((image.shape[1], image.shape[2]), dtype=bool) if image_nodata is None \
                 else np.all(image != image_nodata, axis=0)  
    image[:, ~mask_valid] = np.nan  

    if not np.any(mask_valid):  
        raise ValueError("图像无有效数据像元")  
  
    image_processed = np.zeros_like(image, dtype=np.float32)  
    for i in range(image.shape[0]):  
        band = image[i]  
        valid = band[mask_valid]  
        if len(valid):  
            vmin = np.percentile(valid, 0.5)  
            vmax = np.percentile(valid, 99.5)  
            if vmax - vmin > 0:  
                out = (band - vmin) / (vmax - vmin)  
                out = np.clip(out, 0, 1)  
                out[~mask_valid] = np.nan  
                image_processed[i] = out  
            else:  
                image_processed[i] = np.nan  
        else:  
            image_processed[i] = np.nan  
    image = image_processed  

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
            crs=meta['crs'],  
            transform=meta['transform'],  
            nodata=np.nan  
        ) as dst:  
            dst.write(image)  
            if normalize:  
                dst.write_mask(mask_valid.astype(np.uint8))  

    # 加载并处理标签数据  
    labels = None  
    if label_path:  
        logging.info(f"Loading labels from {label_path}")  
        with rasterio.open(label_path) as src_label:  
            labels = src_label.read(1)  
            label_nodata = src_label.nodata  

        lmask = np.ones_like(labels, dtype=bool) if label_nodata is None else (labels != label_nodata)  
        labels = np.where(lmask, labels, 0).astype(labels.dtype)  
        if output_dir:  
            processed_label_path = os.path.join(output_dir, 'processed_labels.tif')  
            with rasterio.open(  
                processed_label_path,  
                'w',  
                driver='GTiff',  
                height=labels.shape[0],  
                width=labels.shape[1],  
                count=1,  
                dtype=labels.dtype,  
                crs=meta['crs'],  
                transform=meta['transform'],  
                nodata=0  
            ) as dst:  
                dst.write(labels, 1)  
                dst.write_mask(lmask.astype(np.uint8))  

    logging.info(f"影像 shape: {image.shape}, dtype: {image.dtype}")  
    if labels is not None:  
        logging.info(f"标签 shape: {labels.shape}, dtype: {labels.dtype}")  
    if output_dir:  
        logging.info(f"数据已保存到 {output_dir}")  

    return image, labels, meta  
    
def preprocess_and_save_patches(
    image: np.ndarray,  
    labels: np.ndarray,  
    patch_size: int,  
    num_patches: int,  
    save_dir: str,  
    random_seed: Optional[int] = None  
):  
    """  
    根据有效掩码自适应采样 patch 并保存。支持可重复随机性。  
    """  
    if random_seed is not None:  
        random.seed(random_seed)  
        np.random.seed(random_seed)  

    os.makedirs(save_dir, exist_ok=True)  
    images_dir, labels_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')  
    os.makedirs(images_dir, exist_ok=True)  
    os.makedirs(labels_dir, exist_ok=True)  
  
    C, H, W = image.shape  
    saved_patches = 0    # 已经保存的图像块数量  
    attempts = 0         # 采样尝试次数    
    max_attempts = num_patches * 20  # 最大尝试次数，防止无限循环  
  
    while saved_patches < num_patches and attempts < max_attempts:  
        attempts += 1    
        x = random.randint(0, W - patch_size)  
        y = random.randint(0, H - patch_size)  
        # 提取图像块和标签块  
        img_patch = image[:, y:y+patch_size, x:x+patch_size]  # (C, H, W)  
        lbl_patch = labels[y:y+patch_size, x:x+patch_size]  
  
        # 检查图像块中是否存在无效值，跳过包含无效值的图像块 
        if np.isnan(img_patch).any():  
            continue  
        zero_ratio = np.mean(lbl_patch == 0)          # 检查标签块中零值的比例  
        if zero_ratio > 0.2:  
            continue  # 舍弃该图像块，继续下一个采样  
  
        # 转换图像格式 (C, H, W) -> (H, W, C)  
        image_patch_transposed = np.transpose(img_patch, (1, 2, 0))    
        tifffile.imwrite(os.path.join(images_dir, f"image_patch_{saved}.tif"), image_patch_transposed)  
        tifffile.imwrite(os.path.join(labels_dir, f"label_patch_{saved}.tif"), lbl_patch)  
        saved += 1  

    if saved < num_patches:  
        logging.warning(f"最大尝试 {max_attempts} 次，仅采样到 {saved}/{num_patches} 个 patch.")  
    else:  
        logging.info(f"成功采样并保存 {saved} 个 patch.")  

  
if __name__ == '__main__':  
    # ---- 1. 初始化日志 ----  
    setup_logging(log_dir="logs", level=logging.INFO, log_to_console=True)  

    # ---- 2. 加载配置 ----  
    config = get_config()    
    image_path = Path(config['paths']['data']['images']) / config['paths']['input']['train_image']  
    label_path = Path(config['paths']['data']['images']) / config['paths']['input']['train_label']  
  
    # ---- 3. 处理影像与标签 ----  
    image, labels, image_meta = load_and_save_data(  
        image_path=image_path,  
        label_path=label_path,  
        output_dir=config['paths']['data']['process']  
    )  
  
    # ---- 4. 采样切 patch 并保存 ----  
    save_dir = '/home/Dataset/nw/Segmentation/CpeosTest/train_4channel'  
    patch_size = config['dataset']['patch_size']  
    num_patches = config['dataset']['patch_number']  
  
    preprocess_and_save_patches(image, labels, patch_size, num_patches, save_dir)