import numpy as np  
import torch  
import random  
from torch.utils.data import Dataset, DataLoader, random_split  
import random  
from torch.utils.data.distributed import DistributedSampler  


class RemoteSensingDataset(Dataset):  
    """遥感影像数据集"""  
    def __init__(self, image, labels, patch_size, num_patches, preprocess_func = None):  
        """  
        Args:  
            image: 输入图像 [C, H, W]  
            labels: 标签图像 [H, W]  
            patch_size: 图像块大小  
            num_patches: 随机采样的图像块数量  
            preprocess_func: 数据预处理
        """  
        self.image = image  
        self.labels = labels  
        self.patch_size = patch_size  
        self.num_patches = num_patches  
        self.preprocess_func = preprocess_func

        # 确保图像格式正确  
        self.h, self.w = self.image.shape[1:]  # C, H, W  
        
        # 验证输入  
        self._validate_inputs()  

    def _validate_inputs(self):  
        """验证输入数据的有效性"""  
        if self.labels is not None:  
            if self.h != self.labels.shape[0] or self.w != self.labels.shape[1]:  
                raise ValueError("图像和标签尺寸不匹配")  

        if self.h < self.patch_size or self.w < self.patch_size:  
            raise ValueError(f"图像块尺寸{self.patch_size}大于图像尺寸{self.h}x{self.w}")  

    def __len__(self) -> int:  
        return self.num_patches  

    def __getitem__(self, idx):  
        # 随机选择图像块位置  
        x = random.randint(0, self.w - self.patch_size)  
        y = random.randint(0, self.h - self.patch_size)  

        # 提取图像块  
        image_patch = self.image[:3, y:y+self.patch_size, x:x+self.patch_size]  

        # # 应用预处理  
        # if self.preprocess_func:  
        #     image_patch = self.preprocess_func(image_patch)  

        image_patch = torch.from_numpy(image_patch).float()  

        if self.labels is None:  
            return image_patch  

        # 提取并验证标签  
        label_patch = self.labels[y:y+self.patch_size, x:x+self.patch_size]  
        label_patch = torch.from_numpy(label_patch).long()  
        validate_labels(label_patch)  

        return image_patch, label_patch  


def validate_labels(labels: torch.Tensor, num_classes: int = 9) -> None:  
    """验证标签值是否在有效范围内"""  
    unique_labels = torch.unique(labels)  
    min_label = unique_labels.min().item()  
    max_label = unique_labels.max().item()  
    if min_label < 0 or max_label >= num_classes:  
        raise ValueError(f"Labels must be in range [0, {num_classes-1}], "  
                      f"but got range [{min_label}, {max_label}]")  


def create_dataloaders(image, labels, patch_size, num_patches, batch_size,  
                      train_ratio=0.8, num_workers=4, preprocess_func=None):  
    """创建训练和验证数据加载器（单GPU版本）"""  
    # 创建数据集  
    dataset = RemoteSensingDataset(  
        image=image,  
        labels=labels,  
        patch_size=patch_size,  
        num_patches=num_patches,  
        preprocess_func=preprocess_func  
    )  

    # 划分数据集  
    train_size = int(train_ratio * len(dataset))  
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(  
        dataset,   
        [train_size, val_size],  
        generator=torch.Generator().manual_seed(42)  # 保持随机种子以确保可重复性  
    )  
    
    # 创建数据加载器的通用参数  
    loader_kwargs = {  
        'batch_size': batch_size,  
        'num_workers': num_workers,  
        'pin_memory': True,  
        'shuffle': True  # 训练集需要随机打乱  
    }  

    # 创建训练数据加载器  
    train_loader = DataLoader(  
        train_dataset,  
        **loader_kwargs  
    )  
    
    # 验证集不需要随机打乱  
    val_loader = DataLoader(  
        val_dataset,  
        batch_size=batch_size,  
        num_workers=num_workers,  
        pin_memory=True,  
        shuffle=False  
    )  

    return train_loader, val_loader

def split_image_into_patches(image, patch_size=256, overlap=128, preprocess_func=None):  
    """将大图像分割成重叠的小块"""  
    def process_patch(patch):  
        return preprocess_func(patch) if preprocess_func else patch  

    patches = []  
    c, h, w = image.shape  
    stride = patch_size - overlap  

    # 处理常规块和边缘块  
    for y in range(0, h, stride):  
        if y + patch_size > h:  
            y = h - patch_size  
        for x in range(0, w, stride):  
            if x + patch_size > w:  
                x = w - patch_size  
            patch = image[:, y:y+patch_size, x:x+patch_size]  
            patches.append(process_patch(patch))  
            if x + patch_size >= w:  
                break  
        if y + patch_size >= h:  
            break  

    return patches  

def reconstruct_image_from_patches(predictions, image_size, patch_size, overlap):  
    """重建完整的预测图像"""  
    h, w = image_size  
    stride = patch_size - overlap  
    num_classes = predictions[0].shape[0] if predictions else 1  

    # 初始化输出  
    reconstructed = np.zeros((h, w), dtype=np.uint8)  
    confidence = np.zeros((h, w), dtype=np.float32)  

    def update_region(y, x, pred):  
        """更新指定区域的预测结果"""  
        if pred.ndim == 1:  
            pred = pred.reshape(num_classes, patch_size, patch_size)  

        patch_confidence = np.max(pred, axis=0)  
        patch_prediction = np.argmax(pred, axis=0)  

        # 计算有效区域  
        y_end = min(y + patch_size, h)  
        x_end = min(x + patch_size, w)  
        y_range = slice(y, y_end)  
        x_range = slice(x, x_end)  

        # 更新区域  
        current_confidence = confidence[y_range, x_range]  
        update_mask = patch_confidence[:y_end-y, :x_end-x] > current_confidence  
        
        confidence[y_range, x_range][update_mask] = patch_confidence[:y_end-y, :x_end-x][update_mask]  
        reconstructed[y_range, x_range][update_mask] = patch_prediction[:y_end-y, :x_end-x][update_mask]  

    idx = 0  
    for y in range(0, h, stride):  
        if y + patch_size > h:  
            y = h - patch_size  
        for x in range(0, w, stride):  
            if x + patch_size > w:  
                x = w - patch_size  
            if idx < len(predictions):  
                update_region(y, x, predictions[idx])  
                idx += 1  
            if x + patch_size >= w:  
                break  
        if y + patch_size >= h:  
            break  

    return reconstructed  