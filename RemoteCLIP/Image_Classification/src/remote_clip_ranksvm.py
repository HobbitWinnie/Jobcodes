import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

import logging  
import time  
import open_clip  
import torch  
import numpy as np  
import pandas as pd  
from datetime import datetime  
from PIL import Image  
from sklearn.multiclass import OneVsRestClassifier  
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import f1_score, fbeta_score  


# Set up logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

class RemoteCLIPClassifierRankSVM:  
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

        # Initialize the OneVsRestClassifier with a linear SVM  
        self.rank_svm = OneVsRestClassifier(SVC(kernel='linear'))  
        self.scaler = StandardScaler()  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy().astype(np.float32)  

    def train_model(self, dataloader):  
        start_time = time.time()  # Start time for the epoch  
        current_time = datetime.now().strftime('%H:%M:%S')  
        logger.info("RemoteCLIP_RankSVM training start. Time: {}".format(current_time))  

        train_image_features = []  
        train_labels = []  

        with torch.no_grad():  
            for images, labels in dataloader:  
                image_features = self.get_image_features(images.to(self.device))  
                train_image_features.append(image_features)  
                train_labels.extend(labels.numpy().astype(int))

        train_image_features = np.concatenate(train_image_features)  
        train_labels = np.array(train_labels)  

        # Feature scaling  
        train_image_features = self.scaler.fit_transform(train_image_features)  

        # Train RankSVM  
        self.rank_svm.fit(train_image_features, train_labels)  
        
        end_time = time.time()  # End time for the epoch  
        epoch_duration = end_time - start_time  # Calculate the duration 
        logger.info("RemoteCLIP_RankSVM training completed. Time: {:.2f} seconds".format(epoch_duration))  

    def evaluate(self, dataloader):  
        start_time = time.time()
        current_time = datetime.now().strftime('%H:%M:%S')  
        logger.info("RemoteCLIP_RankSVM evaluating start. Time: {}".format(current_time))  

        all_labels = []  
        all_predictions = [] 

        with torch.no_grad():  
            for images, labels in dataloader:  
                images = images.to(self.device)  
                image_features = self.get_image_features(images)  
                image_features = self.scaler.transform(image_features)  
                predictions = self.rank_svm.predict(image_features)  
                
                all_labels.extend(labels.numpy().astype(int))  
                all_predictions.extend(predictions)  
        
        all_labels = np.array(all_labels)  
        all_predictions = np.array(all_predictions)  

        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)  
        f2 = fbeta_score(all_labels, all_predictions, beta=2, average='macro', zero_division=1)  
        
        end_time = time.time()  # End time for the epoch  
        epoch_duration = end_time - start_time  # Calculate the duration 
        logger.info("RemoteCLIP_RankSVM evaluating completed. Time: {:.2f} seconds".format(epoch_duration))  
        
        logger.info(f'F1 Score: {f1}')  
        logger.info(f'F2 Score: {f2}')  

        return f1, f2  
    
    def classify_image(self, image):  
        image = self.preprocess_func(image).unsqueeze(0).to(self.device)  
        image_features = self.get_image_features(image)  
        image_features = self.scaler.transform(image_features)          
        prediction = self.rank_svm.predict(image_features) 
        return prediction  
    
    def classify_images_in_folder(self, folder_path, output_csv):  
        results = []  
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(valid_extensions):  
                try:  
                    image = Image.open(img_path).convert("RGB")  
                    image = self.preprocess_func(image).unsqueeze(0)  
                    top_labels, top_scores = self.classify_image(image)  
                    results.append({"filename": img_name, "top_labels": top_labels, "top_scores": top_scores})  
                except Exception as e:  
                    logger.error(f"处理图像 {img_name} 时出现错误: {e}")  

        if results:  
            output_dir = os.path.dirname(output_csv)  
            if not os.path.exists(output_dir):  
                os.makedirs(output_dir, exist_ok=True)  

            df = pd.DataFrame(results)  
            df.to_csv(output_csv, index=False)  
            logger.info(f"结果已保存到 `{output_csv}`")  
        else:  
            logger.warning("未处理任何有效图像。")