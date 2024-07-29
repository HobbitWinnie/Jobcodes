import os  
import sys  
import torch  
import logging  
from PIL import Image, UnidentifiedImageError    
from pathlib import Path  
from torch.utils.data import DataLoader  

import shutil  # 新增  

# 添加路径以便于导入自定义数据集加载器和分类器  
sys.path.append('/home/nw/Codes/data')  
sys.path.append('/home/nw/Codes/RemoteCLIP/models/image_classification')  

from WHURS19_DatasetLoader import WHURS19DatasetLoader  
from remote_clip_few_shot import RemoteCLIPFewShotClassifier  


# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

scene_classification_labels = [  
    'agricultural_land',  
    'residential_area',  
    'commercial_area',  
    'industrial_area',  
    'water_body',  
    'forest',  
    'desert',  
    'bare_land',  
    'grassland',  
    'airport',  
    'bridge',  
    'port',  
    'transportation_nestwork',  
    'urban_green_space',  
    'power_infrastructure'  
]  


def load_support_dataset(data_path, preprocess_func, num_shots=5):  
    dataset = WHURS19DatasetLoader(data_path, preprocess_func)  
    support_images = []  
    support_labels = []  
    class_sample_count = {cls: 0 for cls in dataset.classes}  

    for image, label, _ in dataset:  
        class_name = dataset.classes[label]  
        if class_sample_count[class_name] < num_shots:  
            support_images.append(image)  
            support_labels.append(class_name)  
            class_sample_count[class_name] += 1  
            
        # Checking if we have enough examples for all classes  
        if all(count >= num_shots for count in class_sample_count.values()):  
            break  

    return support_images, support_labels  

if __name__ == "__main__":  
    ckpt_path = '/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-ViT-L-14.pt'
    support_dataset_path = ''
    model_name = 'ViT-L-14'  
    labels=scene_classification_labels

    # 调用分类函数进行分类  
    query_folder_path = '/mnt/d/nw/Datasets/million-AID/test'  
    output_folder_path = '/mnt/d/nw/million-AID-NW'

    # 构建分类器
    classifier = RemoteCLIPFewShotClassifier(ckpt_path, model_name)  

    # 加载support data
    support_images, support_labels = load_support_dataset(support_dataset_path, classifier.preprocess_func, num_shots=5)  

    classifier.classify_images_in_folder(query_folder_path, classifier, support_images, support_labels, output_folder_path)  