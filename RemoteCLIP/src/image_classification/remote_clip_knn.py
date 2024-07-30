import torch  
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier  
import open_clip  
from PIL import Image  
import pandas as pd  
import os  
from torch.utils.data import DataLoader  

class RemoteCLIPClassifierKNN:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  

        # Load the CLIP model and preprocessing function  
        self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        self.tokenizer = open_clip.get_tokenizer(self.model_name)  

        # Load model checkpoint  
        ckpt = torch.load(ckpt_path, map_location='cpu')  
        self.model.load_state_dict(ckpt)  
        self.model = self.model.to(self.device).eval()  

        self.knn = None  
        self.classes = None  # 用于存储类名  
        self.label_to_index = None  # 用于标签名和索引的映射  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy()  

    def fit_knn(self, dataloader, n_neighbors=20):  
        train_image_features = []  
        train_labels = []  

        for processed_images, labels, _ in dataloader:  
            # Ensure the batch of images is moved to the correct device  
            processed_images = processed_images.to(self.device)  
            features = self.get_image_features(processed_images)  
            train_image_features.append(features)  
            # Append all labels in the current batch  
            train_labels.extend(labels)  

        train_image_features = np.vstack(train_image_features)  
        train_labels = np.array(train_labels)  

        # Getting classes and label indices  
        self.classes = sorted(set(train_labels))  
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}  
        train_label_indices = np.array([self.label_to_index[label] for label in train_labels])  

        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree')  
        self.knn.fit(train_image_features, train_label_indices)  

    def classify_image(self, query_image):  
        query_image = self.preprocess_func(query_image).unsqueeze(0).to(self.device)  
        query_image_features = self.get_image_features(query_image)  

        # Retrieve neighbors and their distances  
        distances, indices = self.knn.kneighbors(query_image_features, n_neighbors=self.knn.n_neighbors)  
        distances, indices = distances[0], indices[0]  

        # Calculate label probabilities  
        label_counts = {label: 0 for label in self.classes}  
        for idx in indices:  
            label = self.classes[self.knn._y[idx]]  
            label_counts[label] += 1  

        label_probabilities = {label: count / self.knn.n_neighbors for label, count in label_counts.items()}  
        sorted_labels = sorted(label_probabilities.items(), key=lambda item: item[1], reverse=True)  

        # Return top 3 labels with their probabilities  
        top_3_labels = sorted_labels[:3]  
        return top_3_labels  

    def classify_images_in_folder(self, folder_path, output_csv):  
        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                image = Image.open(img_path).convert("RGB")  
                top_3_labels = self.classify_image(image)  
                results.append({"filename": img_name, "top_3_labels": top_3_labels})  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  
        print(f"Results saved to `{output_csv}`")  