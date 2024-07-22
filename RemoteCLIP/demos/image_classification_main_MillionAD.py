import os  
import sys  
import logging  
from pathlib import Path  
from torch.utils.data import DataLoader  
from PIL import Image, UnidentifiedImageError
import shutil 

# 取消 PIL 图像尺寸限制  
Image.MAX_IMAGE_PIXELS = None  

sys.path.append('/home/nw/Codes/DatasetLoader')  
sys.path.append('/home/nw/Codes/RemoteCLIP/classifiers')  # assuming this is where the classifier files are  

from WHURS19_DatasetLoader import WHURS19DatasetLoader  
from MillionAID_Dataset_Loader import MillionAIDDatasetLoader  

from remote_clip_classifier_knn import RemoteCLIPClassifierKNN  
from remote_clip_classifier_svm import RemoteCLIPClassifierSVM  
from remote_clip_classifier_rf import RemoteCLIPClassifierRF  

# 定义类标签列表  
LABELS_8_CLASSES=[
    "agriculture land","commercial land","industrial land",
    "public service land","residential land",
    "transportation land","unutilized land","water area"
    ]

LABELS_16_CLASSES=[
    "arable land","grassland","woodland","commercial area",
    "factory area","mining area","power station","sports land",
    "detached house","airport area","highway area","port area",
    "railway area","bare land","lake","river"]

ALL_CLASSES = [
    'agriculture_land', 'airport_area', 'apartment', 'apron', 
    'arable_land', 'bare_land', 'baseball_field', 'basketball_court', 
    'beach', 'bridge', 'cemetery', 'church', 'commercial_area', 
    'commercial_land', 'dam', 'desert', 'detached_house', 'dry_field', 
    'factory_area', 'forest', 'golf_course', 'grassland', 'greenhouse',
    'ground_track_field', 'helipad', 'highway_area', 'ice_land', 
    'industrial_land', 'intersection', 'island', 'lake', 'leisure_land', 
    'meadow', 'mine', 'mining_area', 'mobile_home_park', 'oil_field', 
    'orchard', 'paddy_field', 'parking_lot', 'pier', 'port_area', 
    'power_station', 'public_service_land', 'quarry', 'railway', 
    'railway_area', 'religious_land', 'residential_land', 'river', 
    'road', 'rock_land', 'roundabout', 'runway', 'solar_power_plant', 
    'sparse_shrub_land', 'special_land', 'sports_land', 'stadium', 
    'storage_tank', 'substation', 'swimming_pool', 'tennis_court', 
    'terraced_field', 'train_station', 'transportation_land', 
    'unutilized_land', 'viaduct', 'wastewater_plant', 'water_area', 
    'wind_turbine', 'woodland', 'works'
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

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

def evaluate_classifier(classifier, dataset, labels):  
    total_images = 0  
    correct_predictions = 0  

    for image, ground_truth_label, image_path in dataset:  
        ground_truth_label = labels[ground_truth_label]  # 使用标签索引将其映射回标签  
        
        predicted_index = classifier.classify_image(image) 
        predicted_label = labels[predicted_index]
        if predicted_label == ground_truth_label:  
            correct_predictions += 1  
        total_images += 1  

    accuracy = correct_predictions / total_images if total_images > 0 else 0  
    logging.info(f"Total images: {total_images}\n"  
                 f"Correct predictions: {correct_predictions}\n"  
                 f"Accuracy: {accuracy:.2%}")  

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


def classify_images_in_folder(folder_path, classifier, labels, output_path = None):  # 增加 labels 参数  
    classified_images = []  

    if not os.path.exists(folder_path):  
        print(f"Folder path {folder_path} does not exist.")  
        return  

    for image_path in Path(folder_path).rglob('*.*'):  
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:  
            continue  

        file_name = os.path.basename(image_path)  
        query_image = Image.open(image_path).convert('RGB')  
        query_image = classifier.preprocess_func(query_image)  

        predicted_label_index = classifier.classify_image(query_image)  
        predicted_label = labels[predicted_label_index]  # 确保 labels 是传递进来的参数  
  
        logging.info(f"\nFile: {file_name}\n"  
                     f"Path: {image_path}\n"  
                     f"Predicted label: {predicted_label}\n"  
                     f"{'-'*40}") 
        classified_images.append((image_path, predicted_label))  

    organize_images(classified_images, labels=labels, output_path=output_path)

            
def main(data_path, ckpt_path, query_folder_path, batch_size=32, model_type='svm', num_workers=4, labels=None, output_path = None):  
    if model_type == 'knn':  
        classifier = RemoteCLIPClassifierKNN(ckpt_path=ckpt_path)  
    elif model_type == 'svm':  
        classifier = RemoteCLIPClassifierSVM(ckpt_path=ckpt_path)  
    elif model_type == 'rf':  
        classifier = RemoteCLIPClassifierRF(ckpt_path=ckpt_path)  
    else:  
        raise ValueError("Unsupported model type. Choose 'knn', 'svm', or 'rf'.")  

    millionAIDD_dataset = WHURS19DatasetLoader(data_path=data_path, preprocess_func=classifier.preprocess_func)  
    dataloader = DataLoader(millionAIDD_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  

    if model_type == 'knn':  
        classifier.fit_knn(dataloader, n_neighbors=20)  
    elif model_type == 'svm':  
        classifier.fit_svm(dataloader, C=1.0, kernel='linear')  
    elif model_type == 'rf':  
        classifier.fit_rf(dataloader, n_estimators=100, max_depth=None)  

    classify_images_in_folder(query_folder_path, classifier, labels, output_path )  
    # evaluate_classifier(classifier, millionAIDD_dataset, labels)  


if __name__ == "__main__":  
    data_path = '/mnt/d/nw/million-AID-NW' 
    ckpt_path = '/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-ViT-L-14.pt'  
    # query_folder_path = '/home/nw/Codes/RemoteCLIP/assets/GF2_Selected_50'  

    # 调用分类函数进行分类  
    query_folder_path = '/mnt/d/nw/Datasets/GF2_Data/data'  
    output_folder_path = '/mnt/d/nw/Datasets/GF2_Data/rf_results/LABELS_17_Scene_CLASSES'

    labels = scene_classification_labels  
    main(data_path, ckpt_path, query_folder_path, model_type='rf', labels=labels, output_path=output_folder_path)  # 可替换 'svm' 为 'knn' 或 'rf'  
