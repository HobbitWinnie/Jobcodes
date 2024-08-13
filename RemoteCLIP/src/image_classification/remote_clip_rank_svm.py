import torch  
import numpy as np  
import logging  
import os  
import pandas as pd  
from PIL import Image  
from sklearn.multiclass import OneVsRestClassifier  
from sklearn.svm import SVC  
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler  
from sklearn.model_selection import GridSearchCV  
import open_clip  

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

        self.rank_svm = OneVsRestClassifier(SVC(kernel='linear'))  
        self.mlb = MultiLabelBinarizer()  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy().astype(np.float32)  

    def fit_rank_svm(self, dataloader):  
        train_image_features = []  
        train_labels = []  
        with torch.no_grad():  
            for images, labels, _ in dataloader:  
                image_features = self.get_image_features(images)  
                train_image_features.append(image_features)  
                train_labels.extend(labels.numpy())  

        train_image_features = np.concatenate(train_image_features)  
        train_labels = np.array(train_labels)  

        # Create label list  
        train_labels_list = [np.where(row == 1)[0].tolist() for row in train_labels]  

        # Multi-label Binarization  
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

    def classify_image(self, query_image, top_k=3):  
        query_image = query_image.to(self.device)  
        query_image_features = self.get_image_features(query_image)  

        # Get decision scores from the trained SVM model  
        scores = self.rank_svm.decision_function(query_image_features)  # Shape should be (1, num_classes)  

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
