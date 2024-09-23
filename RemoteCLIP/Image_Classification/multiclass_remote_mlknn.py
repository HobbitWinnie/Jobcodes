import os  
import torch  
import time  
import sys
import logging  
import open_clip  
import numpy as np  
import pandas as pd  
from PIL import Image  
from pathlib import Path  
from datetime import datetime  
from skmultilearn.adapt import MLkNN  
from torch.utils.data import DataLoader  
from torchvision import transforms  
from concurrent.futures import ThreadPoolExecutor, as_completed  
from sklearn.model_selection import train_test_split  

sys.path.append('/home/nw/Codes/data_loader')  
from MLRSNet_loader import MLRSNetDataset


# 设置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  

class RemoteCLIPClassifierMLKNN:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None, n_neighbors=10, s=1.0):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  

        try:  
            # 加载CLIP模型和预处理函数  
            self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
            self.tokenizer = open_clip.get_tokenizer(self.model_name)  

            # 加载模型检查点  
            ckpt = torch.load(ckpt_path, map_location='cpu')  
            self.model.load_state_dict(ckpt)  
            self.model = self.model.to(self.device).eval()  
        except Exception as e:  
            logger.error(f"加载模型失败: {e}")  
            raise e  

        self.mlknn = MLkNN(k=n_neighbors, s=s)  
        self.label_list = [  
            "Forest", "Grassland", "Farmland", "Commercial Area",  
            "Industrial Area", "Mining Area", "Power Station", "Residential Area",  
            "Airport", "Road", "Port Area", "Bridge",  
            "Train Station", "Viaduct", "Bare Land", "Open Water", "Small Water Body"  
        ]  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy()  

    def fit_knn(self, dataloader):  
        start_time = time.time()  
        logger.info("开始训练 RemoteCLIP_MLKNN.")  

        train_image_features = []  
        train_labels = []  

        for images, labels in dataloader:  
            images = images.to(self.device)  
            features = self.get_image_features(images)  
            train_image_features.append(features)  
            train_labels.extend(labels.numpy().astype(int))  

        train_image_features = np.vstack(train_image_features)  
        train_labels = np.array(train_labels)  

        self.mlknn.fit(train_image_features, train_labels)  

        elapsed_time = time.time() - start_time  
        logger.info(f"RemoteCLIP_MLKNN 训练完成. 耗时: {elapsed_time:.2f} 秒")  

    def classify_image(self, query_image):  
        query_image = query_image.to(self.device)  
        query_image_features = self.get_image_features(query_image)  

        predicted = self.mlknn.predict(query_image_features)  
        predicted_labels = predicted.toarray()[0]  

        # label_names = [self.label_list[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1]  

        return predicted_labels  

    def classify_images_in_folder(self, folder_path, output_csv, max_workers=8):  
        if not os.path.exists(folder_path):  
            logger.error(f"文件夹路径 {folder_path} 不存在。")  
            return  

        image_files = list(Path(folder_path).rglob('*.[jJpP][pPnNeEgG]*[gG]'))  
        results = []  

        with ThreadPoolExecutor(max_workers=max_workers) as executor:  
            futures = {executor.submit(self.process_image, image_file): image_file for image_file in image_files}  

            for future in as_completed(futures):  
                try:  
                    image_file, predicted_labels = future.result()  
                    if predicted_labels is not None:  
                        row_result = [os.path.basename(image_file)] + list(predicted_labels)  
                        results.append(row_result)  
                except Exception as e:  
                    logger.error(f"处理文件 {image_file} 时出错: {e}")  

        # Define the headers  
        headers = ["image"] + self.label_list  
        df = pd.DataFrame(results, columns=headers)  
        df.to_csv(output_csv, index=False)  
        logger.info(f"结果已保存到 `{output_csv}`")  

    def process_image(self, image_file):  
        try:  
            with Image.open(image_file) as img:  
                img = img.convert('RGB')  
                img_tensor = self.preprocess_func(img).unsqueeze(0)  
                predicted_labels = self.classify_image(img_tensor)  
            return image_file, predicted_labels  
        except Exception as e:  
            logger.warning(f"处理图像 {image_file} 失败: {e}")  
            return image_file, None
        
def load_MLRSNet_data(images_dir, labels_dir):  
    """加载所有图像和标签数据"""  
    data = []  
    for label_file in os.listdir(labels_dir):  
        if label_file.endswith('.csv'):  
            label_path = os.path.join(labels_dir, label_file)  
            label_data = pd.read_csv(label_path)  
            
            for _, row in label_data.iterrows():  
                image_name = row.iloc[0]  
                labels = row.iloc[1:].values.astype('float')  
                image_path = os.path.join(images_dir, image_name)  
                
                if os.path.exists(image_path):  
                    data.append((image_path, labels))  
                else:  
                    print(f"Warning: Image {image_name} not found in {images_dir}")      
    return data  


if __name__ == "__main__":  
    # 定义路径  
    ckpt_path = '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-RN50.pt'  
    query_folder_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_512_dataset/Images'  
    labels_output_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_512_dataset/Labels_mlknn'  

    train_image_dir = '/home/Dataset/nw/GF2_Data/MultiLabel_dataset/Images'
    train_labels_dir = '/home/Dataset/nw/GF2_Data/MultiLabel_dataset/Lables'

    os.makedirs(labels_output_path, exist_ok=True)  

    # 加载数据  
    data = load_MLRSNet_data(train_image_dir, train_labels_dir)  

    # 划分数据集  
    train_data, test_data = train_test_split(data, test_size=0.001, random_state=42) 

    model_name = 'RN50'  # 'RN50', 'ViT-B-32' 或 'ViT-L-14'  
    classifier = RemoteCLIPClassifierMLKNN(ckpt_path, model_name)  

    train_dataset = MLRSNetDataset(train_data, classifier.preprocess_func)  
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)  

    # 训练模型  
    classifier.fit_knn(train_loader)

    # 分类文件夹中的图像  
    for subfolder in Path(query_folder_path).iterdir():  
        if subfolder.is_dir():  
            csv_output_path = os.path.join(labels_output_path, f"{subfolder.name}.csv")  
            classifier.classify_images_in_folder(subfolder, csv_output_path)  
