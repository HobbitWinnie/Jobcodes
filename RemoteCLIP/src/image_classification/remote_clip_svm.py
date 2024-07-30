import torch  
import numpy as np  
from sklearn.svm import SVC  
import open_clip  
from PIL import Image  
import pandas as pd  
import os   

class RemoteCLIPClassifierSVM:  
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

        self.svm = None  
        self.classes = None  # 用于存储类名  
        self.label_to_index = None  # 用于标签名和索引的映射  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy()  

    def fit_svm(self, dataloader, C=1.0, kernel='linear'):  
        train_image_features = []  
        train_labels = []  
        for processed_images, labels, _ in dataloader:  
            features = self.get_image_features(processed_images)  
            train_image_features.append(features)  
            train_labels.extend(labels)  # 无需转换为 numpy 数组  
        train_image_features = np.vstack(train_image_features)  
        train_labels = np.array(train_labels)  

        # 获取类名和标签到索引的映射  
        self.classes = sorted(set(train_labels))  
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}  
        train_label_indices = np.array([self.label_to_index[label] for label in train_labels])  

        self.svm = SVC(C=C, kernel=kernel, probability=True)  
        self.svm.fit(train_image_features, train_label_indices)  

    def classify_image(self, query_image):  
        query_image = query_image.to(self.device)  
        query_image_features = self.get_image_features(query_image)  

        # 获取预测概率  
        probabilities = self.svm.predict_proba(query_image_features)[0]  
        sorted_indices = np.argsort(probabilities)[::-1]  # 按概率排序  
        top_3_indices = sorted_indices[:3]  

        # 获取对应的标签及概率  
        top_3_labels = [(self.classes[i], probabilities[i]) for i in top_3_indices]  
        return top_3_labels  
    
    def classify_images_in_folder(self, folder_path, output_csv):  
        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                image = Image.open(img_path).convert("RGB")  
                image = self.preprocess_func(image).unsqueeze(0)  
                top_3_labels = self.classify_image(image)  
                results.append({"filename": img_name, "top_3_labels": top_3_labels})  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  
        print(f"Results saved to `{output_csv}`")  
