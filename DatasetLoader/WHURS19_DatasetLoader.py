import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 数据集类
class WHURS19DatasetLoader(Dataset):
    def __init__(self, data_path, preprocess_func):
        self.preprocess_func = preprocess_func
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            class_dir = os.path.join(data_path, cls)
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

        return image, label, image_path
