import os  
import sys  
import torch  
import numpy as np  
import logging  
import open_clip  
from PIL import Image  
from pathlib import Path  
from torch.utils.data import DataLoader  
from sklearn.svm import SVC  

sys.path.append('/home/nw/Codes/DatasetLoader')  
from WHURS19_DatasetLoader import WHURS19DatasetLoader  

# 定义新的标签列表  
LABELS_16_CLASSES = [  
    "arable land","grassland","woodland","commercial area","factory area",
    "mining area","power station","sports land","detached house",
    "airport area","highway area","port area",
    "railway area","bare land","lake","river"
]  

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  


class RemoteCLIPClassifier:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  

        # 加载CLIP模型和预处理函数  
        self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        self.tokenizer = open_clip.get_tokenizer(self.model_name)  

        # 加载模型检查点  
        ckpt = torch.load(ckpt_path, map_location='cpu')  
        self.model.load_state_dict(ckpt)  
        self.model = self.model.to(self.device).eval()  

        self.svc = None  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy()  

    def fit_svc(self, dataloader, C=1.0, kernel='linear'):  
        train_image_features = []  
        train_labels = []  
        for images, labels, paths in dataloader:  
            features = self.get_image_features(images)  
            train_image_features.append(features)  
            train_labels.extend(labels.numpy())  
        train_image_features = np.vstack(train_image_features)  

        train_labels = np.array(train_labels)  # 直接转换为数组  

        self.svc = SVC(C=C, kernel=kernel, probability=True)  
        self.svc.fit(train_image_features, train_labels)  

    def classify_image(self, query_image):  
        query_image = query_image.unsqueeze(0).to(self.device)  
        query_image_features = self.get_image_features(query_image)  

        predicted_label_index = self.svc.predict(query_image_features)[0]  
        predicted_label = labels[predicted_label_index]  

        return predicted_label  

def evaluate_classifier(classifier, dataset):  
    total_images = 0  
    correct_predictions = 0  

    for image, ground_truth_label, _ in dataset:  
        ground_truth_label = labels[ground_truth_label]  # 使用标签索引将其映射回标签  

        predicted_label = classifier.classify_image(image)  
        if predicted_label == ground_truth_label:  
            correct_predictions += 1  
        total_images += 1  

    accuracy = correct_predictions / total_images if total_images > 0 else 0  
    logging.info(f"Total images: {total_images}\n"  
                 f"Correct predictions: {correct_predictions}\n"  
                 f"Accuracy: {accuracy:.2%}")  

def classify_images_in_folder(folder_path, classifier):  
    if not os.path.exists(folder_path):  
        print(f"Folder path {folder_path} does not exist.")  
        return  

    for image_path in Path(folder_path).rglob('*.*'):  
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:  
            continue  

        file_name = os.path.basename(image_path)  
        query_image = Image.open(image_path).convert('RGB')  
        query_image = classifier.preprocess_func(query_image)  

        # 确保 query_image 的形状匹配  
        predicted_label = classifier.classify_image(query_image)  
        logging.info(f"\nFile: {file_name}\n"  
                     f"Path: {image_path}\n"  
                     f"Predicted label: {predicted_label}\n"  
                     f"{'-'*40}")  

def main(data_path, ckpt_path, query_folder_path, batch_size=32, num_workers=4):  
    classifier = RemoteCLIPClassifier(ckpt_path=ckpt_path)  

    WHURS19_dataset = WHURS19DatasetLoader(data_path=data_path, preprocess_func=classifier.preprocess_func)  
    dataloader = DataLoader(WHURS19_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  

    classifier.fit_svc(dataloader, C=1.0, kernel='linear')  

    classify_images_in_folder(query_folder_path, classifier)

    evaluate_classifier(classifier, WHURS19_dataset)  

if __name__ == "__main__":  
    data_path ='/mnt/d/nw/Datasets/Classification-12/WHU-RS19'  
    ckpt_path = '/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-ViT-L-14.pt'  
    query_folder_path = '/home/nw/Codes/RemoteCLIP/assets/GF2_Selected_50'  

    main(data_path, ckpt_path, query_folder_path) 
