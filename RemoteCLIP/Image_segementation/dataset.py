import torch  
import numpy as np  
import random  
from PIL import Image  
from torchvision import transforms  
from torch.utils.data import Dataset, DataLoader, random_split  
from torch.utils.data import Dataset  


class RemoteSensingDataset(Dataset):  
    """遥感影像数据集"""  
    def __init__(self, image, labels, patch_size, num_patches, preprocess_func=None):  
        """  
        Args:  
            image: 输入图像，形状为 [C, H, W]，dtype=float32  
            labels: 标签图像，形状为 [H, W]  
            patch_size: 图像块大小  
            num_patches: 随机采样的图像块数量  
            preprocess_func: 数据预处理函数  
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
        MAX_ATTEMPTS = 20  # 最大尝试次数  

        for attempt in range(MAX_ATTEMPTS):  
            # 随机选择图像块位置  
            x = random.randint(0, self.w - self.patch_size)  
            y = random.randint(0, self.h - self.patch_size)  

            # 提取图像块，形状为 (C, patch_size, patch_size)  
            image_patch = self.image[:, y:y+self.patch_size, x:x+self.patch_size]  
            # 将形状转换为 (patch_size, patch_size, C)  
            image_patch = np.transpose(image_patch, (1, 2, 0))  
            
            # # 确保数据为 RGB 图像（取前3个通道）  
            # if image_patch.shape[2] > 3:  
            #     image_patch = image_patch[:, :, :3]  

            # # 确定数据的数值范围并进行转换  
            # max_value = image_patch.max()  
            # if max_value <= 1.0:  
            #     # 数据范围在 [0,1]，转换为 [0,255]  
            #     image_patch = (image_patch * 255).astype('uint8')  
            # else:  
            #     # 数据范围在 [0,255]  
            #     image_patch = image_patch.astype('uint8')  

            # 将 NumPy 数组转换为 PIL.Image 对象  
            image_patch = Image.fromarray(image_patch, mode='RGB')  

            # 应用数据增强  
            if self.preprocess_func is not None:  
                image_patch = self.preprocess_func(image_patch)  
                # 预处理后应得到 Tensor，形状为 (C, H, W)  
                if not isinstance(image_patch, torch.Tensor):  
                    # 如果预处理后仍是 PIL.Image，需转换为 Tensor  
                    image_patch = transforms.ToTensor()(image_patch)  
            else:  
                # 如果没有提供预处理函数，直接转换为 Tensor  
                image_patch = transforms.ToTensor()(image_patch)  

            if self.labels is not None:  
                # 提取标签patch  
                label_patch = self.labels[y:y+self.patch_size, x:x+self.patch_size]  

                # 计算0像素值的比例  
                zero_ratio = (label_patch == 0).sum() / (self.patch_size * self.patch_size)  

                # 如果比例小于30%，则返回该patch  
                if zero_ratio < 0.3:  
                    label_patch = torch.from_numpy(label_patch).long()  
                    validate_labels(label_patch)  
                    return image_patch, label_patch  
            else:  
                return image_patch  

        # 如果达到最大尝试次数仍未找到合适的patch，则返回最后一次尝试的patch  
        if self.labels is not None:  
            # 重新提取 label_patch，防止变量未定义  
            label_patch = self.labels[y:y+self.patch_size, x:x+self.patch_size]  
            label_patch = torch.from_numpy(label_patch).long()  
            validate_labels(label_patch)  
            return image_patch, label_patch  
        else:  
            return image_patch  

def validate_labels(labels: torch.Tensor, num_classes: int = 9) -> None:  
    """验证标签值是否在有效范围内"""  
    unique_labels = torch.unique(labels)  
    min_label = unique_labels.min().item()  
    max_label = unique_labels.max().item()  
    if min_label < 0 or max_label >= num_classes:  
        raise ValueError(f"Labels must be in range [0, {num_classes-1}], but got range [{min_label}, {max_label}]")

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