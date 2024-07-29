import torch  
import numpy as np  
import pandas as pd  
import os  
from sklearn.neighbors import NearestNeighbors  
from sklearn.preprocessing import MultiLabelBinarizer  
from PIL import Image  
import open_clip  

class RemoteCLIPClassifierMLKNN:  
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
        self.mlb = MultiLabelBinarizer()  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy()  

    def fit_knn(self, dataloader, n_neighbors=20):  
        train_image_features = []  
        train_labels = []  
        
        for images, labels, paths in dataloader:  
            features = self.get_image_features(images)  
            train_image_features.append(features)  
            train_labels.extend(labels.numpy())  
        
        train_image_features = np.vstack(train_image_features)  
        train_labels = np.array(train_labels)  

        # Multi-label Binarization  
        self.mlb.fit(train_labels)  
        binarized_labels = self.mlb.transform(train_labels)  

        self.knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')  
        self.knn.fit(train_image_features, binarized_labels)  

    def classify_image(self, query_image, k=20, s=1.0):  
        query_image = query_image.unsqueeze(0).to(self.device)  
        query_image_features = self.get_image_features(query_image)  

        distances, indices = self.knn.kneighbors(query_image_features, n_neighbors=k)  
        neighbor_labels = self.knn.predict(indices)  

        # Calculate the probabilities for each label  
        label_counts = np.sum(neighbor_labels, axis=1)  
        label_proba = (label_counts + s) / (k + 2 * s)  
        
        # Binarize the probabilities (label is present if probability > 0.5)  
        predicted_labels = (label_proba > 0.5).astype(int)  

        return self.mlb.inverse_transform(predicted_labels)  
    
    def classify_images_in_folder(self, folder_path, output_csv, k=20, s=1.0):  
        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                image = Image.open(img_path).convert("RGB")  
                image = self.preprocess_func(image).unsqueeze(0)  
                labels = self.classify_image(image, k, s)  
                results.append({"filename": img_name, "labels": labels})  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  