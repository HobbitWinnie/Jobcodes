import os  
import sys  
import torch  
import logging  
from PIL import Image, UnidentifiedImageError    
from pathlib import Path  
from torch.utils.data import DataLoader  

import shutil  # 新增  

# 添加路径以便于导入自定义数据集加载器和分类器  
sys.path.append('/home/nw/Codes/DatasetLoader')  
sys.path.append('/home/nw/Codes/RemoteCLIP/classifiers')  # assuming this is where the classifier files are  
from WHURS19_DatasetLoader import WHURS19DatasetLoader  
from remote_clip_few_shot_classifier import RemoteCLIPFewShotClassifier  



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
    'transportation_network',  
    'urban_green_space',  
    'power_infrastructure'  
]  

def organize_image(image_path, predicted_label, output_path):  
    """  
    将单张图像移动到对应的标签文件夹。  

    参数：  
    image_path (str): 图像的路径  
    predicted_label (str): 图像的预测标签  
    output_path (str): 输出图像文件夹的路径  
    """  
    file_name = os.path.basename(image_path)  
    label_folder_path = os.path.join(output_path, predicted_label)  
    destination_path = os.path.join(label_folder_path, file_name)  
    
    # 创建目标文件夹（如果不存在）  
    os.makedirs(label_folder_path, exist_ok=True)  
    
    try:  
        shutil.copy(str(image_path), destination_path)  
        if os.path.exists(destination_path):  
            logging.info(f"Copied {file_name} to {destination_path} successfully.")  
        else:  
            logging.error(f"Failed to copy {file_name} to {destination_path}.")  
    except Exception as e:  
        logging.error(f"Error copying file {file_name} to {destination_path}: {e}")  

def classify_images_in_folder(folder_path, classifier, support_data, output_path):  
    """  
    对文件夹中的所有图像进行分类，并将分类结果返回。  

    参数：  
    folder_path (str): 查询图像文件夹的路径  
    classifier (RemoteCLIPZeroShotClassifier): 分类器实例  
    output_path (str): 输出图像文件夹的路径  
    """  
    if not os.path.exists(folder_path):  
        logging.error(f"Folder path {folder_path} does not exist.")  
        return  
        
    # 创建主输出文件夹  
    os.makedirs(output_path, exist_ok=True)  
    
    for image_path in Path(folder_path).rglob('*.*'):  
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:  
            continue  
        
        file_name = os.path.basename(image_path)  
        try:  
            # 尝试打开图像文件  
            query_image = Image.open(image_path).convert('RGB')  
            query_image = classifier.preprocess_func(query_image)  

            predicted_label, probability = classifier.few_shot_classify(support_data, query_image)  
            logging.info(f"File: {file_name}\n"  
                         f"Path: {image_path}\n"  
                         f"Predicted label: {predicted_label}\n"  
                         f"Probability: {probability:.2%}\n"  
                         f"{'-'*40}")  
            if probability > 0.5:
                organize_image(image_path, predicted_label, output_path)  
        
        except UnidentifiedImageError:  
            logging.error(f"Cannot identify image file {file_name}. Skipping.")  
        except IOError:  
            logging.error(f"Cannot open image file {file_name}. It might be corrupted. Skipping.")  
        except Exception as e:  
            logging.error(f"Error processing file {file_name}: {e}")  

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

    # 构建分类器
    classifier = RemoteCLIPFewShotClassifier(ckpt_path, model_name)  

    # 加载support data
    support_images, support_labels = load_support_dataset(support_dataset_path, classifier.preprocess_func, num_shots=5)  

    # 调用分类函数进行分类  
    query_folder_path = '/mnt/d/nw/Datasets/million-AID/test'  
    output_folder_path = '/mnt/d/nw/million-AID-NW'
    classify_images_in_folder(query_folder_path, classifier, support_images, support_labels, output_folder_path)  