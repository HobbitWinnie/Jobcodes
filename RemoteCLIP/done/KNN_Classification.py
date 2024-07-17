import os
from PIL import Image
from pathlib import Path  
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import open_clip
import logging  

LABEL_MAPPING = {
    0: 'Airport',
    1: 'Beach',
    2: 'Bridge',
    3: 'Commercial',
    4: 'Desert',
    5: 'Farmland',
    6: 'Forest',
    7: 'Industrial',
    8: 'Meadow',
    9: 'Mountain',
    10: 'Park',
    11: 'Parking',
    12: 'Pond',
    13: 'Port',
    14: 'Residential',
    15: 'River',
    16: 'Viaduct',
    17: 'footballField',
    18: 'railwayStation'
}

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

# 数据集类
class WHURS19DatasetLoader(Dataset):
    def __init__(self, root_dir, preprocess_func):
        self.root_dir = root_dir
        self.preprocess_func = preprocess_func

        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.image_paths.append(os.path.join(class_dir, fname))
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess_func(image)
        return image, label

# 分类器类
class RemoteCLIPClassifier:
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess_func, _ = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device).eval()

        self.label_encoder = LabelEncoder()
        self.knn = None

    def get_image_features(self, images):
        images = images.to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def fit_knn(self, dataloader, n_neighbors=20):
        train_image_features = []
        train_labels = []
        for images, labels in dataloader:
            features = self.get_image_features(images)
            train_image_features.append(features)
            train_labels.extend(labels.numpy())
        train_image_features = np.vstack(train_image_features)
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)

        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree')
        self.knn.fit(train_image_features, train_labels_encoded)

    def classify_image(self, query_image):
        query_image = query_image.unsqueeze(0).to(self.device)
        query_image_features = self.get_image_features(query_image)
        predicted_label_encoded = self.knn.predict(query_image_features)
        predicted_label = self.label_encoder.inverse_transform(predicted_label_encoded)
        return predicted_label[0]


# 定义图像分类函数  
def classify_images_in_folder(folder_path, classifier, label_mapping):  
    # 检查输入文件夹路径是否存在  
    if not os.path.exists(folder_path):  
        print(f"Folder path {folder_path} does not exist.")  
        return  
    
    # 遍历文件夹下所有图像文件  
    for image_path in Path(folder_path).rglob('*.*'):  
        # 检查文件扩展名  
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:  
            continue 

        # 获取文件名  
        file_name = os.path.basename(image_path)  

        # 打开图像并进行预处理  
        query_image = Image.open(image_path).convert('RGB')  
        query_image = classifier.preprocess_func(query_image)  # 使用模型预处理函数  

        # 对图像进行分类  
        predicted_label = classifier.classify_image(query_image)  
        output_label = label_mapping.get(predicted_label, "Unknown Label")  
        
        # 打印分类结果  
        # 打印分类结果  
        logging.info(f"\nFile: {file_name}\n"  
                     f"Path: {image_path}\n"  
                     f"Predicted label: {output_label}\n"  
                     f"{'-'*40}")  

def main(root_dir, ckpt_path, query_folder_path, batch_size=32, n_neighbors=20, num_workers=4):  
    # 创建分类器实例  
    classifier = RemoteCLIPClassifier(ckpt_path=ckpt_path)  

    # 创建数据集和数据加载器  
    dataset = WHURS19DatasetLoader(root_dir=root_dir, preprocess_func=classifier.preprocess_func)  
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  

    # 训练KNN分类器  
    classifier.fit_knn(dataloader, n_neighbors=n_neighbors)  

    # 对查询图像进行分类  
    classify_images_in_folder(query_folder_path, classifier, LABEL_MAPPING) 


if __name__ == "__main__":

    # 数据集根目录和RemoteCLIP检查点路径
    root_dir = '/mnt/d/nw/Datasets/Classification-12/WHU-RS19'
    ckpt_path = '/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-ViT-L-14.pt'
    query_folder_path = '/mnt/d/nw/Datasets/Classification-12/testdata'

    main(root_dir, ckpt_path, query_folder_path)  
