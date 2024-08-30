import os  
import torch  
import numpy as np  
import pandas as pd  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import MultiLabelBinarizer  
from PIL import Image  
import open_clip  
from sklearn.metrics import f1_score  
import logging  

# 设置日志记录  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

class RemoteCLIPClassifierMLKNN:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  
        
        # 加载CLIP模型和预处理函数  
        self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        
        # 加载模型检查点  
        try:  
            ckpt = torch.load(ckpt_path, map_location='cpu')  
            self.model.load_state_dict(ckpt)  
            logger.info("成功加载模型检查点。")  
        except Exception as e:  
            logger.error(f"加载模型检查点时出错：{e}")  
            raise  

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

        for images, labels in dataloader:  
            features = self.get_image_features(images)  
            train_image_features.append(features)  
            train_labels.extend(labels.numpy())  # 确保标签在CPU上并为numpy格式  

        train_image_features = np.vstack(train_image_features)  
        train_labels = np.array(train_labels)  

        # 多标签二值化  
        binarized_labels = self.mlb.fit_transform(train_labels)  
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree')  
        self.knn.fit(train_image_features, binarized_labels)  

    def classify_image(self, query_image, k=20, s=1.0):  
        query_image_features = self.get_image_features(query_image)  

        distances, indices = self.knn.kneighbors(query_image_features, n_neighbors=k)  
        neighbor_labels = self.knn.predict(query_image_features)  

        # 计算每个标签的概率  
        label_counts = np.sum(neighbor_labels, axis=0)  
        label_proba = (label_counts + s) / (k + 2 * s)  

        # 二值化概率（如果概率 > 0.5，则标签存在）  
        predicted_labels = (label_proba > 0.5).astype(int)  

        return self.mlb.inverse_transform(predicted_labels.reshape(1, -1))[0]  

    def evaluate(self, dataloader, k=20, s=1.0):  
        self.model.eval()  
        all_labels = []  
        all_predictions = []  

        with torch.no_grad():  
            for images, labels in dataloader:  
                images = images.to(self.device)  
                predicted_labels_batch = []  

                # 对批次中的每个图像进行分类  
                for img in images:  
                    predicted_labels = self.classify_image(img.unsqueeze(0), k, s)  
                    # 将预测的标签转换为多标签数组格式  
                    predicted_array = np.zeros(len(self.mlb.classes_))  
                    for label in predicted_labels:  
                        if label in self.mlb.classes_:  
                            index = self.mlb.classes_.tolist().index(label)  
                            predicted_array[index] = 1  
                    predicted_labels_batch.append(predicted_array)  
                
                all_labels.extend(labels.cpu().numpy())  # 确保标签在CPU上并为numpy格式  
                all_predictions.extend(predicted_labels_batch)  

        # 评估的多标签二值化  
        binarized_labels = np.array(all_labels)  
        binarized_predictions = np.array(all_predictions)  

        # 确保标签和预测的形状一致  
        if binarized_labels.shape != binarized_predictions.shape:  
            raise ValueError("标签和预测的形状不一致。")  

        # 计算F1分数  
        f1 = f1_score(binarized_labels, binarized_predictions, average='macro', zero_division=1)  
        logger.info(f'F1 Score: {f1}')  
        return f1

    def classify_images_in_folder(self, folder_path, output_csv, k=20, s=1.0):  
        results = []  
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(valid_extensions):  
                try:  
                    image = Image.open(img_path).convert("RGB")  
                    image_tensor = self.preprocess_func(image).unsqueeze(0).to(self.device)  
                    labels = self.classify_image(image_tensor, k, s)  
                    results.append({"filename": img_name, "labels": labels})  
                except Exception as e:  
                    logger.error(f"处理图像 {img_name} 时出错：{e}")  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  
        logger.info(f"结果已保存到 `{output_csv}`")