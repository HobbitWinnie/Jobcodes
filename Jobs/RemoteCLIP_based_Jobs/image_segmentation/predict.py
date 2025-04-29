import sys  
sys.path.append('/home/nw/Codes')  

import torch  
import logging  
import os  
import rasterio  
import json  
import numpy as np  
from datetime import datetime  
from torch.cuda.amp import autocast  
from tqdm import tqdm  
from pathlib import Path  

from utils.set_logging import setup_logging  
from .config.config import get_config  
from .data.dataset import load_and_save_data, split_image_into_patches, reconstruct_image_from_patches, CustomTransform  
from Models.RemoteCLIP_based_Segmentation.factory import segmentation_model_factory  


def load_model(config, checkpoint_path, device):
    """加载训练好的模型"""
    # 初始化模型  
    model = segmentation_model_factory(  
        model_type=config['model']['model_type'],  
        model_name=config['model']['model_name'],  
        ckpt_path=config['paths']['model'].get('clip_ckpt', None),  
        num_classes=config['dataset']['num_classes'],  
        dropout_rate=config.get('dropout_rate', 0.2),  
        use_aux_loss=config.get('use_aux_loss', True),  
        initial_features=config.get('initial_features', 128),  
        device_ids=config.get('device_ids', [0])  
    )  

    # 检查权重文件是否存在  
    if not checkpoint_path.is_file():  
        raise FileNotFoundError(f"未找到模型权重: {checkpoint_path}")  

    logging.info(f"尝试加载模型权重: {checkpoint_path}")      
    checkpoint = torch.load(checkpoint_path, map_location=device)  
    state_dict = checkpoint.get('model_state_dict', checkpoint)  

    # 自动去除module前缀  
    new_state_dict = {}  
    for k, v in state_dict.items():  
        new_key = k.replace('module.', '') if k.startswith('module.') else k  
        new_state_dict[new_key] = v  

    # 加载权重  
    model_state_keys = set(model.state_dict().keys())  
    checkpoint_keys = set(new_state_dict.keys())  
    missing_keys = model_state_keys - checkpoint_keys  
    unused_keys = checkpoint_keys - model_state_keys  
    if missing_keys:  
        logging.warning(f"模型中缺少以下权重: {missing_keys}")  
    if unused_keys:  
        logging.warning(f"权重文件中存在未使用的权重: {unused_keys}")  

    try:  
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)  
        logging.info("模型权重加载成功")  
    except RuntimeError as e:  
        logging.error(f"模型权重加载失败: {str(e)}")  
        raise e  

    model.eval()  
    return model  


def predict_single_image(
        model, 
        image_path, 
        patch_size, 
        overlap, 
        device, 
        preprocess_func, 
        output_path=None
    ):
    """
    对单张图像进行预测并保存  
    """
    logging.info(f"加载图像: {image_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

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
        nodata_mask = (original_data[0] == nodata_value) if nodata_value is not None else np.zeros_like(original_data[0], dtype=bool)

    # 分割图像为patches
    patches = split_image_into_patches(test_image, patch_size, overlap, preprocess_func)

    # 预测
    predictions = []
    model.eval()
    with torch.no_grad():
        for patch in tqdm(patches, desc="Processing patches"):
            patch_tensor = patch.clone().detach().unsqueeze(0).to(device)  
            try:
                with autocast():
                    output = model(patch_tensor)
                    pred = output['main'].argmax(1) if isinstance(output, dict) else output.argmax(1)  
                predictions.append(pred)
            except Exception as e:
                logging.error(f"Error processing patch: {str(e)}")
                continue

    # 重建完整图像
    reconstructed_prediction = reconstruct_image_from_patches(
        predictions,
        (test_image.shape[1], test_image.shape[2]),
        patch_size,
        overlap
    )

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


def predict_images(
        model, 
        image_paths, 
        output_paths, 
        patch_size, 
        overlap, 
        device, 
        preprocess_func
    ):
    """预测多张图像"""
    assert len(image_paths) == len(output_paths)  
    for img_path, out_path in zip(image_paths, output_paths):
        if not os.path.exists(img_path):
            logging.error(f"Image file not found: {img_path}")
            continue
        try:
            predict_single_image(
                model=model,
                image_path=img_path,
                patch_size=patch_size,
                overlap=overlap,
                device=device,
                preprocess_func=preprocess_func,
                output_path=out_path
            )
        except Exception as e:
            logging.error(f"Error predicting {img_path}: {str(e)}")

def setup_environment(config):  
    # 日志和实验目录创建  
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  
    exp_dir = Path(config['paths']['model']['save_dir']) / timestamp  
    exp_dir.mkdir(parents=True, exist_ok=True)  
    setup_logging(exp_dir / 'predict.log')  
    with open(exp_dir / 'config.json', 'w') as f:  
        json.dump(config.config, f, indent=4)  
    return exp_dir

def main():
    """主程序入口"""
    try:
        # 初始化配置、日志、实验目录  
        config = get_config()
        exp_dir = setup_environment(config)  

        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logging.info(f"CUDA版本: {torch.version.cuda}")
            logging.info(f"可用GPU: {torch.cuda.get_device_name(0)}")

        # 1. 加载模型
        checkpoint_path = Path(config['paths']['model']['best_model'])
        model = load_model(config, checkpoint_path, device)

        # 2. 输入、输出路径自动组装  
        image_dirs = [
            os.path.join(config['paths']['data']['images'], config['paths']['input']['train_mask']),
            os.path.join(config['paths']['data']['images'], config['paths']['input']['test_image'])
        ]

        output_dirs = [
            os.path.join(config['paths']['data']['results'], config['paths']['output']['train_mask_result']),
            os.path.join(config['paths']['data']['results'], config['paths']['output']['test_image_result'])
        ]

        # 3. 如果自定义预处理transform支持config，可在此传递  
        preprocess_func = CustomTransform()  

        # 4. 批量预测
        predict_images(
            model=model,
            image_paths=image_dirs,
            output_paths=output_dirs,
            patch_size=config['dataset']['patch_size'],
            overlap=config['predict']['overlap'],
            device=device,
            preprocess_func=preprocess_func
        )

        logging.info("预测完成！结果已保存到输出目录。")
    except KeyboardInterrupt:  
        logging.warning("中断推理流程，安全退出。")  
    except Exception as e:
        logging.error(f"程序出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()

