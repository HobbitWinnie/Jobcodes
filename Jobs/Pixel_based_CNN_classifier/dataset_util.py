
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  
from remote_data_loader import GeoTIFFLoader, PatchSampler

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
    patch_size: int = 7,
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


def get_dataloaders(
        patches,
        labels,
        batch_size=192,  
        test_size=0.2,          # 更合理的默认划分比例  
        num_workers=8,          # 根据CPU核心数优化  
        pin_memory=True,        # 提升GPU传输效率  
        persistent_workers=True # 保持worker进程  
    ):
    
    X_train, X_val, y_train, y_val= train_test_split(
        patches, labels,
        test_size=test_size, 
        random_state=42,
    ) 

    # 创建数据集  
    train_dataset = RemoteSensingDataset(X_train, y_train)  
    val_dataset = RemoteSensingDataset(X_val, y_val) 

    # 打印数据集的样本数量  
    print(f"Training dataset size: {len(train_dataset)}")  
    print(f"Testing dataset size: {len(val_dataset)}")  

    
    # 配置数据加载器  
    train_loader = DataLoader(  
        train_dataset,  
        batch_size=batch_size,  
        shuffle=True,  
        num_workers=num_workers,  
        pin_memory=pin_memory,  
        persistent_workers=persistent_workers,  
        prefetch_factor=2    # 提升数据预取  
    )  
    
    val_loader = DataLoader(  
        val_dataset,  
        batch_size=batch_size,  
        shuffle=False,       # 验证集不需要shuffle  
        num_workers=num_workers//2,  # 减少验证集workers  
        pin_memory=pin_memory,  
        persistent_workers=persistent_workers  
    ) 

    # 打印数据集统计信息  
    print(f"\n{' Dataset Info ':-^40}")  
    print(f"| {'Split':<15} | {'Samples':>8} |")  
    print(f"| {'-'*15} | {'-'*8} |")  
    print(f"| {'Training':<15} | {len(train_dataset):>8} |")  
    print(f"| {'Validation':<15} | {len(val_dataset):>8} |")  
    print(f"{'-'*40}\n")  

    return train_loader, val_loader  
