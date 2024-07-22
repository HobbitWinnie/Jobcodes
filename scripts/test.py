import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import open_clip

# 数据集类
class WHURS19Dataset(Dataset):
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
class WHURS19RemoteCLIPClassifier:
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
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def fit_knn(self, dataloader):
        train_image_features = []
        train_labels = []
        for images, labels in dataloader:
            features = self.get_image_features(images)
            train_image_features.append(features)
            train_labels.extend(labels.numpy())
        train_image_features = np.vstack(train_image_features)
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)

        self.knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        self.knn.fit(train_image_features, train_labels_encoded)

    def classify_image(self, query_image):
        query_image = query_image.unsqueeze(0).to(self.device)
        query_image_features = self.get_image_features(query_image)
        predicted_label_encoded = self.knn.predict(query_image_features)
        predicted_label = self.label_encoder.inverse_transform(predicted_label_encoded)
        return predicted_label[0]

# 主程序
if __name__ == "__main__":
    # 数据集根目录和RemoteCLIP模型路径
    root_dir = '/path/to/wrs19/'  # 请确保该路径正确
    ckpt_path = '/path/to/RemoteCLIP-ViT-L-14.pt'  # 确保此路径正确

    # 创建分类器实例
    classifier = WHURS19RemoteCLIPClassifier(ckpt_path=ckpt_path)

    # 创建数据集和数据加载器，使用模型的预处理函数
    dataset = WHURS19Dataset(root_dir=root_dir, preprocess_func=classifier.preprocess_func)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 训练KNN分类器
    classifier.fit_knn(dataloader)

    # 分类查询图像示例
    query_image_path = 'query_image.jpg'  # 确保查询图像路径正确
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = classifier.preprocess_func(query_image)  # 使用模型的预处理函数

    # 对查询图像进行分类
    predicted_label = classifier.classify_image(query_image)
    print(f'Predicted label: {predicted_label}')