import os  
import sys  
import logging  
from pathlib import Path  
from torch.utils.data import DataLoader  
from PIL import Image  

sys.path.append('/home/nw/Codes/DatasetLoader')  
sys.path.append('/home/nw/Codes/RemoteCLIP/classifiers')  # assuming this is where the classifier files are  

from WHURS19_DatasetLoader import WHURS19DatasetLoader  
from remote_clip_classifier_knn import RemoteCLIPClassifierKNN  
from remote_clip_classifier_svm import RemoteCLIPClassifierSVM  
from remote_clip_classifier_rf import RemoteCLIPClassifierRF  

# 定义16类标签列表  
LABELS_16_CLASSES = [  
    "arable land", "grassland", "woodland", "commercial area", "factory area",  
    "mining area", "power station", "sports land", "detached house",  
    "airport area", "highway area", "port area",  
    "railway area", "bare land", "lake", "river"  
]  

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

def evaluate_classifier(classifier, dataset):  
    total_images = 0  
    correct_predictions = 0  

    for image, ground_truth_label, image_path in dataset:  
        ground_truth_label = LABELS_16_CLASSES[ground_truth_label]  # 使用标签索引将其映射回标签  
        predicted_label = classifier.classify_image(image)  
        if predicted_label == ground_truth_label:  
            correct_predictions += 1  
        total_images += 1  

    accuracy = correct_predictions / total_images if total_images > 0 else 0  
    logging.info(f"Total images: {total_images}\n"  
                 f"Correct predictions: {correct_predictions}\n"  
                 f"Accuracy: {accuracy:.2%}")  

def classify_images_in_folder(folder_path, classifier):  
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
        predicted_label = LABELS_16_CLASSES[predicted_label_index]   
  
        logging.info(f"\nFile: {file_name}\n"  
                     f"Path: {image_path}\n"  
                     f"Predicted label: {predicted_label}\n"  
                     f"{'-'*40}")  

def main(data_path, ckpt_path, query_folder_path, batch_size=32, model_type='svm', num_workers=4):  
    if model_type == 'knn':  
        classifier = RemoteCLIPClassifierKNN(ckpt_path=ckpt_path)  
    elif model_type == 'svm':  
        classifier = RemoteCLIPClassifierSVM(ckpt_path=ckpt_path)  
    elif model_type == 'rf':  
        classifier = RemoteCLIPClassifierRF(ckpt_path=ckpt_path)  
    else:  
        raise ValueError("Unsupported model type. Choose 'knn', 'svm', or 'rf'.")  

    whurs19_dataset = WHURS19DatasetLoader(data_path=data_path, preprocess_func=classifier.preprocess_func)  
    dataloader = DataLoader(whurs19_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  

    if model_type == 'knn':  
        classifier.fit_knn(dataloader, n_neighbors=20)  
    elif model_type == 'svm':  
        classifier.fit_svm(dataloader, C=1.0, kernel='linear')  
    elif model_type == 'rf':  
        classifier.fit_rf(dataloader, n_estimators=100, max_depth=None)  

    classify_images_in_folder(query_folder_path, classifier)  
    evaluate_classifier(classifier, whurs19_dataset)  

if __name__ == "__main__":  
    data_path ='/mnt/d/nw/Datasets/Classification-12/WHU-RS19'  
    ckpt_path = '/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-ViT-L-14.pt'  
    query_folder_path = '/home/nw/Codes/RemoteCLIP/assets/GF2_Selected_50'  

    main(data_path, ckpt_path, query_folder_path, model_type='svm')  # 可替换 'svm' 为 'knn' 或 'rf'