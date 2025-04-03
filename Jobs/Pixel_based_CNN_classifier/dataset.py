
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from data_loader import GeoTIFFLoader, PatchSampler

logger = logging.getLogger(__name__)

class RemoteSensingDataset(Dataset):
    """PyTorch遥感数据集"""
    
    def __init__(
        self, 
        image_patches: np.ndarray, 
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Args:
            image_patches: 影像块数据 [N, C, H, W]
            labels: 对应标签 [N]
            transform: 数据增强方法
        """
        self.image_patches = image_patches
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.image_patches[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return (
            torch.as_tensor(sample, dtype=torch.float32),
            torch.as_tensor(label, dtype=torch.long)
        )

class DatasetManager:
    """数据集存储管理"""
    
    @staticmethod
    def save_dataset(X: np.ndarray, y: np.ndarray, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "X_samples.npy", X)
        np.save(save_dir / "y_samples.npy", y)
        logger.info(f"数据集已保存至 {save_dir}")

    @staticmethod
    def load_dataset(load_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            X = np.load(load_dir / "X_samples.npy")
            y = np.load(load_dir / "y_samples.npy")
            logger.info(f"从 {load_dir} 加载数据集")
            return X, y
        except FileNotFoundError:
            logger.warning(f"{load_dir} 中未找到数据集")
            return None

def prepare_dataset(
    image_path: Path,
    label_path: Path,
    save_dir: Path,
    sample_size: int = 50000,
    patch_size: int = 7
) -> Tuple[np.ndarray, np.ndarray, float]:
    """端到端数据集准备流程
    Returns:
        X: 样本数据 [N, C, H, W]
        y: 标签数据 [N]
        nodata: 无效值标识
    """
    # 加载原始数据
    image, meta, nodata = GeoTIFFLoader.load_geotiff(image_path)
    labels, _, _ = GeoTIFFLoader.load_geotiff(label_path)
    
    # 尝试加载已存数据集
    saved_data = DatasetManager.load_dataset(save_dir)
    if saved_data:
        return saved_data[0], saved_data[1], nodata
    
    # 创建采样器并采样
    sampler = PatchSampler(patch_size)
    X, y = sampler.sample(image, labels[0], nodata, sample_size)  # 标签为单通道
    
    # 保存数据集
    DatasetManager.save_dataset(X, y, save_dir)
    return X, y, nodata