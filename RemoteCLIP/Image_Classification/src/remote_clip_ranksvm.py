import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"  # 指定显卡  

import torch  
import numpy as np  
import logging  
import pandas as pd  
from PIL import Image  
from sklearn.multiclass import OneVsRestClassifier  
from sklearn.svm import SVC  
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler  
from sklearn.model_selection import GridSearchCV  
import open_clip  
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
        self.mlb = MultiLabelBinarizer()  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy().astype(np.float32)  

    def train_model(self, dataloader):  
        train_image_features = []  
        train_labels = []  
        with torch.no_grad():  
            for images, labels in dataloader:  
                image_features = self.get_image_features(images.to(self.device))  
                train_image_features.append(image_features)  
                train_labels.extend(labels.numpy())  

        train_image_features = np.concatenate(train_image_features)  
        train_labels = np.array(train_labels)  

        # Create label list and binarize it  
        train_labels_list = [np.where(row == 1)[0].tolist() for row in train_labels]  
        binarized_labels = self.mlb.fit_transform(train_labels_list)  

        # Feature scaling  
        scaler = StandardScaler()  
        train_image_features = scaler.fit_transform(train_image_features)  

        logger.info(f"Features dtype: {train_image_features.dtype}")  

        # Hyperparameter tuning using GridSearchCV  
        param_grid = {'estimator__C': [0.1, 1.0, 10]}  
        grid = GridSearchCV(self.rank_svm, param_grid, cv=5, n_jobs=-1)  
        grid.fit(train_image_features, binarized_labels)  
        self.rank_svm = grid.best_estimator_  

    def evaluate(self, dataloader, top_k=3):  
        self.model.eval()  
        all_labels = []  
        all_predictions = []  

        with torch.no_grad():  
            for images, labels in dataloader:  
                images = self.preprocess_func(images).to(self.device)  
                batch_labels = [np.where(label.numpy() == 1)[0].tolist() for label in labels]  
                all_labels.extend(batch_labels)  

                for image in images:  
                    top_labels, _ = self.classify_image(image.unsqueeze(0), top_k)  
                    predicted_indices = [self.mlb.classes_.tolist().index(label) for label in top_labels]  
                    all_predictions.append(predicted_indices)  

        # Binarize predictions and true labels  
        binarized_labels = self.mlb.transform(all_labels)  
        binarized_predictions = self.mlb.transform(all_predictions)  

        # Calculate F1 score  
        f1 = f1_score(binarized_labels, binarized_predictions, average='macro', zero_division=1)  
        logger.info(f'F1 Score: {f1}')  

        # Calculate F2 score  
        f2 = fbeta_score(binarized_labels, binarized_predictions, beta=2, average='macro', zero_division=1)  
        logger.info(f'F2 Score: {f2}')  

        return f1, f2  
    
    
    def classify_image(self, query_image, top_k=3):  
        query_image = query_image.to(self.device)  
        query_image_features = self.get_image_features(query_image)  

        # Get decision scores from the trained SVM model  
        scores = self.rank_svm.decision_function(query_image_features)  

        # Handle scores to get top_k classes  
        top_indices = np.argsort(scores[0])[::-1][:top_k]  
        top_labels = [self.mlb.classes_[i] for i in top_indices]  
        top_scores = scores[0][top_indices]  

        return top_labels, top_scores  

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