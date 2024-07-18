import torch  
import numpy as np  
from sklearn.ensemble import RandomForestClassifier  
import open_clip  


class RemoteCLIPClassifierRF:  
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

        self.rf = None  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy()  

    def fit_rf(self, dataloader, n_estimators=100, max_depth=None):  
        train_image_features = []  
        train_labels = []  
        for images, labels, paths in dataloader:  
            features = self.get_image_features(images)  
            train_image_features.append(features)  
            train_labels.extend(labels.numpy())  
        train_image_features = np.vstack(train_image_features)  
        train_labels = np.array(train_labels)  

        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)  
        self.rf.fit(train_image_features, train_labels)  

    def classify_image(self, query_image):  
        query_image = query_image.unsqueeze(0).to(self.device)  
        query_image_features = self.get_image_features(query_image)  
        predicted_label_index = self.rf.predict(query_image_features)[0]  
        return predicted_label_index