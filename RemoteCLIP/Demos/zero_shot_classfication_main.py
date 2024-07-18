import os  
import sys  
import torch  
import logging  
from PIL import Image  
from pathlib import Path  
from torch.utils.data import DataLoader  

import shutil  # 新增  

# 添加路径以便于导入自定义数据集加载器和分类器  
sys.path.append('/home/nw/Codes/DatasetLoader')  
sys.path.append('/home/nw/Codes/RemoteCLIP/classifiers')  # assuming this is where the classifier files are  
from WHURS19_DatasetLoader import WHURS19DatasetLoader  
from remote_clip_zero_shot_classifier import RemoteCLIPZeroShotClassifier  


# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

LABELS_16_CLASSES = [  
    "arable land", "grassland", "woodland", "commercial area", "factory area",  
    "mining area", "power station", "sports land", "detached house",  
    "airport area", "highway area", "port area",  
    "railway area", "bare land", "lake", "river"  
]  

LABELS_8_CLASSES=[
    "agriculture land","commercial land","industrial land",
    "public service land","residential land",
    "transportation land","unutilized land","water area"
    ]


def organize_images(classified_images, output_path, labels):  
    """  
    为分类结果中的每个标签生成文件夹，并将图像移动到对应的标签文件夹。  
    
    参数：  
    classified_images (list): 包含图像路径和其预测标签的列表  
    folder_path (str): 查询图像文件夹的路径  
    """  
    # 检查并创建文件夹  
    if not os.path.exists(output_path):  
        os.makedirs(output_path)  

    # 创建标签文件夹  
    for label in labels:  
        label_folder_path = os.path.join(output_path, label)  
        os.makedirs(label_folder_path, exist_ok=True)  
    
    # 移动图像到对应的标签文件夹  
    for image_path, predicted_label in classified_images:  
        file_name = os.path.basename(image_path)  
        destination_folder = os.path.join(output_path, predicted_label)  
        destination_path = os.path.join(destination_folder, file_name)  
        shutil.copy(str(image_path), destination_path) 


def classify_images_in_folder(folder_path, classifier, output_path, labels):  
    """  
    对文件夹中的所有图像进行分类，并将分类结果返回。  
    
    参数：  
    folder_path (str): 查询图像文件夹的路径  
    classifier (RemoteCLIPZeroShotClassifier): 分类器实例  
    output_path (str): 输出图像文件夹的路径  
    """  
    classified_images = []  

    if not os.path.exists(folder_path):  
        logging.error(f"Folder path {folder_path} does not exist.")  
        return  

    for image_path in Path(folder_path).rglob('*.*'):  
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:  
            continue  
        file_name = os.path.basename(image_path)  
        query_image = Image.open(image_path).convert('RGB')  
        query_image = classifier.preprocess_func(query_image)  

        predicted_label, probability = classifier.classify_image(query_image)  
        logging.info(f"File: {file_name}\n"  
                     f"Path: {image_path}\n"  
                     f"Predicted label: {predicted_label}\n"  
                     f"Probability: {probability:.2%}\n"  
                     f"{'-'*40}")  
        
        classified_images.append((image_path, predicted_label))  

    organize_images(classified_images, output_path, labels)
        

def evaluate_classifier(classifier, dataset_loader, labels):  
    total_images = 0  
    correct_predictions = 0  

    for image, ground_truth_label, image_path in dataset_loader:  
        ground_truth_label = labels[ground_truth_label]  

        predicted_label, probability = classifier.classify_image(image)  
        if predicted_label != ground_truth_label:  
            logging.info(f"image_path: {image_path}\n"  
                         f"Predicted label: {predicted_label}\n"  
                         f"Probability: {probability:.2%}\n"  
                         f"Ground truth label: {ground_truth_label}\n"  
                         f"{'-'*40}")  

        if predicted_label == ground_truth_label:  
            correct_predictions += 1  
        total_images += 1  

    accuracy = correct_predictions / total_images if total_images > 0 else 0  
    logging.info(f"Total images: {total_images}\n"  
                 f"Correct predictions: {correct_predictions}\n"  
                 f"Accuracy: {accuracy:.2%}")  

if __name__ == "__main__":  
    ckpt_path = '/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-ViT-L-14.pt'  
    model_name = 'ViT-L-14'  
    labels=LABELS_16_CLASSES

    # 构建分类器
    classifier = RemoteCLIPZeroShotClassifier(ckpt_path, model_name, labels=labels)  

    # 调用分类函数进行分类  
    query_folder_path = '/mnt/d/nw/Datasets/GF2_Data/data'  
    output_folder_path = '/mnt/d/nw/Datasets/GF2_Data/LABELS_16_CLASSES'
    classify_images_in_folder(query_folder_path, classifier, output_folder_path,labels)  