from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np  



# 分层平衡采样策略实现（超参数β设置为10）  
class HierarchicalBalancedSampler(Sampler):  
    def __init__(self, labels, num_samples_per_class=10):  
        self.labels = labels  
        self.num_samples_per_class = num_samples_per_class  
        self.class_indices = self._get_class_indices()  

    def _get_class_indices(self):  
        class_indices = {}  
        for idx, label in enumerate(self.labels):  
            label = int(label)  
            if label not in class_indices:  
                class_indices[label] = []  
            class_indices[label].append(idx)  
        return class_indices  

    def __iter__(self):  
        indices = []  
        for label, idx_list in self.class_indices.items():  
            if len(idx_list) >= self.num_samples_per_class:  
                sampled_indices = np.random.choice(idx_list, self.num_samples_per_class, replace=False)  
            else:  
                sampled_indices = np.random.choice(idx_list, self.num_samples_per_class, replace=True)  
            indices.extend(sampled_indices)  
        np.random.shuffle(indices)  
        return iter(indices)  

    def __len__(self):  
        return self.num_samples_per_class * len(self.class_indices)  



# 定义数据集类
class HsiDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # 数据应为numpy数组，形状为[num_samples, channels, height, width]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    

def create_dataloaders(image_dir, labels_dir, batch_size, train_ratio=0.8, num_workers=4):  
    """创建训练和验证数据加载器（单GPU版本）"""  
    # 创建数据集  
    dataset = HsiDataset(  
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
    sampler = HierarchicalBalancedSampler(labels.flatten(), num_samples_per_class)
    train_loader = DataLoader(  
        train_dataset,  
        **loader_kwargs,
        sampler=sampler
    )  
    
    # 验证集不需要随机打乱  
    val_loader = DataLoader(  
        val_dataset,  
        batch_size=batch_size,  
        num_workers=num_workers,  
        pin_memory=True,  
        shuffle=False  
    )  