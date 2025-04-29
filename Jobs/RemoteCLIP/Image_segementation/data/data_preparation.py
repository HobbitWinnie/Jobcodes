import os  
import random  
import numpy as np  
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
  
def load_and_save_data(image_path, label_path, output_dir, normalize=True):  
    # 创建相应的目录结构  
    if output_dir is not None:  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        output_dir = os.path.join(output_dir, timestamp)  
        os.makedirs(output_dir, exist_ok=True)  
  
    try:  
        # 加载并处理图像数据  
        logging.info(f"正在加载图像: {image_path}")  
        with rasterio.open(image_path) as src:  
            image = src.read()  # 图像形状为 (bands, height, width)  
            image_meta = src.meta.copy()  
            image_nodata = src.nodata
  
        # # 仅保留前三个波段  
        # if image.shape[0] >= 3:  
        #     image = image[:3, :, :]  
        # else:  
        #     raise ValueError("图像的波段数少于 3，无法提取前三个波段")  
  
        # 创建无效值掩码  
        if image_nodata is not None:  
            image_mask = np.all(image != image_nodata, axis=0)  
        else:  
            image_mask = np.full((image.shape[1], image.shape[2]), True, dtype=bool)  
  
        # 处理无效值，将无效值设置为 np.nan  
        image = image.astype(np.float32)  
        image[:, ~image_mask] = np.nan  
  
        # 数据有效性检查  
        if np.all(~image_mask):  
            raise ValueError("No valid data in image")  
  
        image_processed = np.zeros_like(image, dtype=np.float32)  
        for i in range(image.shape[0]):  
            band_data = image[i]  
            valid_data = band_data[image_mask]  
  
            if len(valid_data) > 0:  
                min_val = np.percentile(valid_data, 0.5)  
                max_val = np.percentile(valid_data, 99.5)  
                print(f"波段 {i+1} 的拉伸范围：{min_val} - {max_val}")  
  
                # 拉伸有效数据到 0-1 范围  
                band_scaled = (band_data - min_val) / (max_val - min_val)  
                
                # 将数据裁剪到 0-1 范围内  
                band_scaled = np.clip(band_scaled, 0.0, 1.0)  
  
                # 保留无效值为 np.nan  
                band_scaled[~image_mask] = np.nan  
  
                image_processed[i] = band_scaled  
            else:  
                # 如果没有有效数据，直接复制原始数据  
                image_processed[i] = band_data  
  
        # 图像数据已经转换为 float32 类型  
        image = image_processed  
  
        for i in range(image.shape[0]):  
            valid_band_data = image[i][image_mask]  
            print(f"波段 {i+1} 的拉伸后数据范围：{valid_band_data.min()} - {valid_band_data.max()}")  
  
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
                transform=image_meta['transform'],  
                nodata=np.nan  
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
                labels_nodata = src.nodata  
  
            # 处理标签无效值  
            if labels_nodata is not None:  
                label_mask = labels != labels_nodata  
            else:  
                label_mask = np.full(labels.shape, True, dtype=bool)  
  
            # 替换无效值为 0  
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
                    transform=image_meta['transform'],  
                    nodata=0  
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
  
    return image, labels, image_meta
  
  
def preprocess_and_save_patches(image, labels, patch_size, num_patches, save_dir):  
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
  
    max_attempts = num_patches * 20  # 最大尝试次数，防止无限循环  
  
    while saved_patches < num_patches and attempts < max_attempts:  
        attempts += 1  
  
        x = random.randint(0, w - patch_size)  
        y = random.randint(0, h - patch_size)  
  
        # 提取图像块和标签块  
        image_patch = image[:, y:y+patch_size, x:x+patch_size]  # (C, H, W)  
        label_patch = labels[y:y+patch_size, x:x+patch_size]  
  
        # 检查图像块中是否存在无效值  
        if np.isnan(image_patch).any():  
            continue  # 跳过包含无效值的图像块  
  
        # 检查标签块中零值的比例  
        zero_ratio = np.sum(label_patch == 0.0) / (patch_size * patch_size)  
        if zero_ratio > 0.2:  
            continue  # 舍弃该图像块，继续下一个采样  
  
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
  
    image, labels, image_meta = load_and_save_data(  
        image_path=image_path,  
        label_path=label_path,  
        output_dir=config['paths']['data']['process']  
    )  
  
    # 设置参数并运行预处理  
    save_dir = '/home/Dataset/nw/Segmentation/CpeosTest/train_4channel'  
    patch_size = config['dataset']['patch_size']  
    num_patches = config['dataset']['patch_number']  
  
    preprocess_and_save_patches(image, labels, patch_size, num_patches, save_dir)