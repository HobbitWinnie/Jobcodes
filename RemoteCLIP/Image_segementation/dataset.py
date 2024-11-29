import os  
import torch  
import numpy as np  
import tifffile  
import torchvision.transforms.functional as F  
from torchvision import transforms  
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split  
from torch.utils.data import Dataset  

class CustomTransform:  
    def __init__(self):  
        self.size = 224  
        self.mean = [0.48145466, 0.4578275, 0.40821073]  
        self.std = [0.26862954, 0.26130258, 0.27577711]  
        self.interpolation = transforms.InterpolationMode.BICUBIC  

    def __call__(self, img):  
        # img 可以是 numpy.ndarray 或 torch.Tensor，值域在 0-1 之间  
        # 如果输入是 numpy.ndarray，转换为 Tensor 并调整维度  
        if isinstance(img, np.ndarray):  
            img = torch.from_numpy(img).float()  
            # 调整维度顺序，从 (H, W, C) 到 (C, H, W)  
            img = img.permute(2, 0, 1)  
        # 如果输入是 PIL.Image，转换为 Tensor  
        elif isinstance(img, Image.Image):  
            img = transforms.ToTensor()(img)  
        else:  
            raise TypeError(f"不支持的图像类型：{type(img)}")  

        # 确保图像为 float32 类型  
        img = img.float()  

        # 如果图像是单通道，转换为三通道  
        if img.dim() == 2:  
            img = img.unsqueeze(0).repeat(3, 1, 1)  
        elif img.dim() == 3:  
            if img.size(0) == 1:  # 形状为 (1, H, W)  
                img = img.repeat(3, 1, 1)  
            elif img.size(0) != 3:  
                raise ValueError(f"图像通道数为 {img.size(0)}，无法处理。")  

        # 调整尺寸  
        img = F.resize(img, self.size, self.interpolation)  

        # 中心裁剪  
        img = F.center_crop(img, self.size)  

        # 归一化  
        img = F.normalize(img, mean=self.mean, std=self.std)  

        return img  

class RemoteSensingDataset(Dataset):  
    """遥感影像数据集"""  
    def __init__(self, images_dir, labels_dir, preprocess_func=None):  
        """  
        Args:  
            images_dir: 图像块的目录路径  
            labels_dir: 标签块的目录路径  
            preprocess_func: 图像预处理函数（可选）  
        """  
        self.images_dir = images_dir  
        self.labels_dir = labels_dir  
        self.preprocess_func = CustomTransform()
        
        # 获取所有图像和标签文件名  
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])  
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.tif')])  
        
        # 验证图像和标签数量匹配  
        if len(self.image_files) != len(self.label_files):  
            raise ValueError("图像和标签数量不匹配")  
        
        # 确保图像和标签文件名一一对应  
        for img_file, lbl_file in zip(self.image_files, self.label_files):  
            if img_file.replace('image', '') != lbl_file.replace('label', ''):  
                raise ValueError(f"图像文件 {img_file} 和标签文件 {lbl_file} 不对应")  
        
    def __len__(self) -> int:  
        return len(self.image_files)  

    def __getitem__(self, idx):  
        # 获取图像和标签文件路径  
        image_path = os.path.join(self.images_dir, self.image_files[idx])  
        label_path = os.path.join(self.labels_dir, self.label_files[idx])  
        
        # 使用 tifffile 读取图像和标签  
        image = tifffile.imread(image_path)  # 标签形状： (H, W)  
        label = tifffile.imread(label_path)  # 标签形状： (H, W)  

         # 应用预处理函数（如果提供了）
        if self.preprocess_func is not None:  
            image = self.preprocess_func(image)  
        
        # 转换标签为 Tensor  
        label = torch.from_numpy(label).long()

        # 验证标签值是否在有效范围内（假设类别数为 9）  
        validate_labels(label)  

        return image, label  

def validate_labels(labels: torch.Tensor, num_classes: int = 9) -> None:  
    """验证标签值是否在有效范围内"""  
    unique_labels = torch.unique(labels)  
    min_label = unique_labels.min().item()  
    max_label = unique_labels.max().item()  
    if min_label < 0 or max_label >= num_classes:  
        raise ValueError(f"标签值应在 [0, {num_classes-1}] 范围内，但得到的范围是 [{min_label}, {max_label}]")  
    

def create_dataloaders(image_dir, labels_dir, batch_size, train_ratio=0.8, num_workers=4, preprocess_func=None):  
    """创建训练和验证数据加载器（单GPU版本）"""  
    # 创建数据集  
    dataset = RemoteSensingDataset(  
        images_dir=image_dir,  
        labels_dir=labels_dir,  
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