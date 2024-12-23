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
        self.mean = [0.48145466, 0.4578275, 0.40821073, 0.5]  
        self.std = [0.26862954, 0.26130258, 0.27577711, 0.5]  
        self.interpolation = transforms.InterpolationMode.BICUBIC  

    def __call__(self, img):  
        # img 可以是 numpy.ndarray 或 torch.Tensor，值域在 0-1 之间  
        if isinstance(img, np.ndarray):  
            # 如果输入是 numpy.ndarray，则转换为 Tensor (H, W, C -> C, H, W)  
            img = torch.from_numpy(img).float()  
            img = img.permute(2, 0, 1)  
        elif isinstance(img, Image.Image):  
            # 如果输入是 PIL.Image，则转换为 Tensor  
            img = transforms.ToTensor()(img)  
        else:  
            raise TypeError(f"不支持的图像类型：{type(img)}")  

        # 确保图像为 float32 类型  
        img = img.float()  

        # 如果图像是单通道，扩展为4通道（通常假设任务需求）  
        if img.dim() == 2:  # (H, W)  
            img = img.unsqueeze(0).repeat(4, 1, 1)  # 补充4个通道全相同  
        elif img.dim() == 3:  
            if img.size(0) == 1:  # 单通道 (1, H, W)  
                img = img.repeat(4, 1, 1)  # 重复为4通道  
            elif img.size(0) == 3:  # RGB (3, H, W)  
                # 如果输入是3通道，增加一个全零通道，扩展到4通道  
                extra_channel = torch.zeros((1, img.size(1), img.size(2)))  # (1, H, W)  
                img = torch.cat((img, extra_channel), dim=0)  
            elif img.size(0) != 4:  # 不是4通道  
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
    def __init__(self, images_dir, labels_dir):  
        """  
        Args:  
            images_dir: 图像块的目录路径  
            labels_dir: 标签块的目录路径  
            preprocess_func: 图像预处理函数（可选）  
        """  
        self.images_dir = images_dir  
        self.labels_dir = labels_dir  
        self.preprocess_func = CustomTransform()

        self.class_names = [
            'background',  
            'wheat',  
            'corn',  
            'sunflower',  
            'watermelon',  
            'tomato',  
            'sugar beet',  
            'green onion',  
            'zucchini'  
        ]
        
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

        # 提取类别名称  
        unique_labels = np.unique(label)  
        
        # 创建类别名称映射  
        label_to_classname = {label_value: self.class_names[label_value] for label_value in unique_labels if label_value < len(self.class_names)}  
        text_inputs = ' '.join(label_to_classname[label_value] for label_value in unique_labels if label_value < len(self.class_names))  

        # 验证标签值是否在有效范围内（假设类别数为 9）  
        validate_labels(label)  

        return image, label, text_inputs

def validate_labels(labels: torch.Tensor, num_classes: int = 9) -> None:  
    """验证标签值是否在有效范围内"""  
    unique_labels = torch.unique(labels)  
    min_label = unique_labels.min().item()  
    max_label = unique_labels.max().item()  
    if min_label < 0 or max_label >= num_classes:  
        raise ValueError(f"标签值应在 [0, {num_classes-1}] 范围内，但得到的范围是 [{min_label}, {max_label}]")  
    

def create_dataloaders(image_dir, labels_dir, batch_size, train_ratio=0.8, num_workers=0):  
    """创建训练和验证数据加载器（单GPU版本）"""  
    # 创建数据集  
    dataset = RemoteSensingDataset(  
        images_dir=image_dir,  
        labels_dir=labels_dir,  
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

def reconstruct_image_from_patches(predictions, image_size, patch_size, overlap, confidences=None):  
    """  
    重建完整的预测图像，采用置信度决定重叠区域保留的值。  
    
    Args:  
        predictions (list): 每个 patch 的预测结果 (Tensor)，形状为 (1, patch_size, patch_size)。  
        image_size (tuple): 原始图像的大小 (height, width)。  
        patch_size (int): 每个 patch 的大小。  
        overlap (int): patch 之间的重叠像素数。  
        confidences (list, optional): 每个 patch 的置信度矩阵 (Tensor)，形状为 (patch_size, patch_size)。默认为 None。  
                                      如果为 None，所有 patch 的 confidence 默认值为 1。  

    Returns:  
        np.ndarray: 完整拼接后的预测图像。  
    """  
    h, w = image_size  # 原图尺寸  
    stride = patch_size - overlap  # 滑动窗口步长  

    # 初始化输出图像和置信度图  
    reconstructed = np.zeros((h, w), dtype=np.float32)  # 用于存储拼接后的图像  
    confidence_map = np.zeros((h, w), dtype=np.float32)  # 用于存储当前图像每像素的总置信度  

    # 函数用于更新指定区域  
    def update_region(y, x, pred, patch_confidence):  
        """  
        更新图像中一个区域的值，基于置信度。  
        Args:  
            y (int): 当前 patch 的 y 起始坐标。  
            x (int): 当前 patch 的 x 起始坐标。  
            pred (Tensor): 当前 patch 的预测结果，形状为 (1, patch_size, patch_size)。  
            patch_confidence (Tensor or None): 当前 patch 的置信度矩阵，形状为 (patch_size, patch_size)。  
                                                如果 None，置信度值默认为 1。  
        """  
        if isinstance(pred, torch.Tensor):  # 如果是 Tensor，则转换成 NumPy  
            pred = pred.squeeze(0).cpu().numpy()  

        if patch_confidence is not None and isinstance(patch_confidence, torch.Tensor):  
            patch_confidence = patch_confidence.cpu().numpy()  

        # 如果没有提供置信度，则默认为所有像素的置信度为 1  
        if patch_confidence is None:  
            patch_confidence = np.ones(pred.shape, dtype=np.float32)  

        # 计算有效的范围  
        y_end = min(y + patch_size, h)  
        x_end = min(x + patch_size, w)  
        y_range = slice(y, y_end)  
        x_range = slice(x, x_end)  

        # 处理 patch 内的实际使用区域（边界可能小于 patch_size）  
        pred_y_end = y_end - y  
        pred_x_end = x_end - x  

        # 当前 patch 的有效数据  
        patch_prediction = pred[:pred_y_end, :pred_x_end]  
        patch_conf = patch_confidence[:pred_y_end, :pred_x_end]  

        # 对图像和置信度图进行更新（基于置信度覆盖）  
        current_conf = confidence_map[y_range, x_range]  
        update_mask = patch_conf > current_conf  # 只有置信度更高的像素才会覆盖  
        
        # 更新图像和置信度  
        confidence_map[y_range, x_range][update_mask] = patch_conf[update_mask]  
        reconstructed[y_range, x_range][update_mask] = patch_prediction[update_mask]  

    # 遍历所有 patches 并更新  
    idx = 0  
    for y in range(0, h, stride):  
        if y + patch_size > h:  # 调整 y 位置  
            y = h - patch_size  
        for x in range(0, w, stride):  
            if x + patch_size > w:  # 调整 x 位置  
                x = w - patch_size  
            if idx < len(predictions):  # 确保索引范围有效  
                pred = predictions[idx]  
                patch_conf = confidences[idx] if confidences is not None else None  
                update_region(y, x, pred, patch_conf)  
                idx += 1  
            if x + patch_size >= w:  # 到达图像右边缘  
                break  
        if y + patch_size >= h:  # 到达图像下边缘  
            break  

    # 返回最终生成的图像  
    return reconstructed.astype(np.float32)