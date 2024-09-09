import torch  
import numpy as np  
from skmultilearn.adapt import MLkNN  
from sklearn.metrics import f1_score, fbeta_score  
import open_clip  
from PIL import Image  
import pandas as pd  
import os  

class RemoteCLIPClassifierMLKNN:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None, n_neighbors=10, s=1.0):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  

        # Load the CLIP model and preprocessing function  
        self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        self.tokenizer = open_clip.get_tokenizer(self.model_name)  

        # Load model checkpoint  
        ckpt = torch.load(ckpt_path, map_location='cpu')  
        self.model.load_state_dict(ckpt)  
        self.model = self.model.to(self.device).eval()  

        self.mlknn = MLkNN(k=n_neighbors, s=s)  
        self.label_to_index = None  # Map label names to indices  

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
            # Ensure the batch of images is moved to the correct device  
            images = images.to(self.device)  
            features = self.get_image_features(images)  
            train_image_features.append(features)  
            train_labels.extend(labels.numpy())  # Assuming labels are arrays/lists of label indices  

        train_image_features = np.vstack(train_image_features)  
        train_labels = np.array(train_labels)  

        self.mlknn.fit(train_image_features, train_labels)  


    def evaluate(self, dataloader):  
        all_true_labels = []  
        all_predicted_labels = []  

        for images, labels, in dataloader:  
            images = images.to(self.device)  
            features = self.get_image_features(images)  
            predicted = self.mlknn.predict(features)  

            all_true_labels.extend(labels.numpy())  
            all_predicted_labels.extend(predicted.toarray())  

        all_true_labels = np.array(all_true_labels)  
        all_predicted_labels = np.array(all_predicted_labels)  

        f1 = f1_score(all_true_labels, all_predicted_labels, average='macro', zero_division=1)  
        f2 = fbeta_score(all_true_labels, all_predicted_labels, beta=2, average='macro', zero_division=1)  

        print(f'F1 Score: {f1}')  
        print(f'F2 Score: {f2}')  

        return f1, f2
    
    def classify_image(self, query_image):  
        query_image = self.preprocess_func(query_image).unsqueeze(0).to(self.device)  
        query_image_features = self.get_image_features(query_image)  

        # Predict multi-labels for the given image  
        predicted = self.mlknn.predict(query_image_features)  
        predicted_labels = predicted.toarray()[0]  
        
        # Map indices to label names (if available)  
        label_names = [self.label_to_index[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1]  
        
        return label_names  

    def classify_images_in_folder(self, folder_path, output_csv):  
        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                image = Image.open(img_path).convert("RGB")  
                predicted_labels = self.classify_image(image)  
                results.append({"filename": img_name, "predicted_labels": predicted_labels})  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  
        print(f"Results saved to `{output_csv}`")  

