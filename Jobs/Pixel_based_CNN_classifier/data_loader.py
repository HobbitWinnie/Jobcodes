
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import rasterio

logger = logging.getLogger(__name__)

class GeoTIFFLoader:
    """遥感影像加载工具类"""
    
    @staticmethod
    def load_geotiff(file_path: Path) -> Tuple[np.ndarray, dict, float]:
        """加载GeoTIFF文件
        Args:
            file_path: 文件路径
        Returns:
            data: 影像数据 [C, H, W]
            meta: 元数据
            nodata: 无效值标识
        """
        try:
            with rasterio.open(file_path) as src:
                data = src.read()
                meta = src.meta.copy()
                nodata = src.nodata
            logger.info(f"成功加载 {file_path}: 形状 {data.shape}")
            return data, meta, nodata
        except Exception as e:
            logger.error(f"加载 {file_path} 失败: {str(e)}")
            raise

class PatchSampler:
    """影像块采样器"""
    
    def __init__(self, patch_size: int = 7):
        self.patch_size = patch_size
        self.pad_size = patch_size // 2

    def sample(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        nodata_value: float,
        sample_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """执行随机采样
        Args:
            image: 原始影像 [C, H, W]
            labels: 标签数据 [H, W]
            nodata_value: 无效值
            sample_size: 采样数量
        """
        # 创建有效区域掩膜
        valid_mask = (labels != nodata_value)
        valid_coords = np.argwhere(valid_mask)
        
        # 随机选择采样点
        selected_indices = np.random.choice(
            len(valid_coords), 
            size=min(sample_size, len(valid_coords)),
            replace=False
        )
        selected_coords = valid_coords[selected_indices]

        # 预处理图像填充
        padded_image = np.pad(
            image,
            ((0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size)),
            mode='edge'
        )

        # 提取影像块
        patches = np.stack([
            padded_image[
                :, 
                row:row + self.patch_size, 
                col:col + self.patch_size
            ] for row, col in selected_coords
        ])
        
        return patches.astype(np.float32), labels[valid_mask][selected_indices].astype(np.int64)
