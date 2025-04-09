import os  
import torch  
from torch.utils.data import Dataset  
import pandas as pd  
from PIL import Image  
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader  


class MLRSNetDataset(Dataset):  
    def __init__(self, data, preprocess_func):  
        """  
        初始化MultiLabelDataset类。  
        
        :param data: 包含图像路径和标签的列表  
        :param preprocess_func: 图像预处理函数  
        """  
        self.data = data  
        self.preprocess_func = preprocess_func  

    def __len__(self):  
        """返回数据集中样本的数量"""  
        return len(self.data)  

    def __getitem__(self, idx):  
        """  
        根据索引获取样本。  
        
        :param idx: 数据集中样本的索引  
        :return: 预处理后的图像张量和标签张量  
        """  
        img_path, labels = self.data[idx]  
        image = Image.open(img_path).convert('RGB')  
        image_tensor = self.preprocess_func(image)  
        
        return image_tensor, torch.tensor(labels, dtype=torch.float32)  
    
def load_MLRSNet_data(images_dir, labels_dir):  
    """加载所有图像和标签数据"""  
    data = []  
    for label_file in os.listdir(labels_dir):  
        if label_file.endswith('.csv'):  
            label_path = os.path.join(labels_dir, label_file)  
            label_data = pd.read_csv(label_path)  
            category = label_file.replace('.csv', '')  
            image_folder = os.path.join(images_dir, category)  
            
            for _, row in label_data.iterrows():  
                image_name = row.iloc[0]  
                labels = row.iloc[1:].values.astype('float')  
                image_path = os.path.join(image_folder, image_name)  
                
                if os.path.exists(image_path):  
                    data.append((image_path, labels))  
                else:  
                    print(f"Warning: Image {image_name} not found in {image_folder}")      
    return data  


def get_dataloaders(
        images_dir,  
        labels_dir,  
        preprocess,  
        batch_size=192,  
        test_size=0.2,          # 更合理的默认划分比例  
        num_workers=8,          # 根据CPU核心数优化  
        pin_memory=True,        # 提升GPU传输效率  
        persistent_workers=True # 保持worker进程  
    ):
    
    # 加载原始数据  
    full_data = load_MLRSNet_data(images_dir, labels_dir)  

    # 划分数据集  
    train_data, test_data = train_test_split(
        full_data, 
        test_size=test_size, 
        random_state=42,
    ) 

    # 创建训练和测试数据集  
    train_dataset = MLRSNetDataset(train_data, preprocess)  
    val_dataset = MLRSNetDataset(test_data, preprocess)  

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