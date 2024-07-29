import os  
import sys  
import torch  
import open_clip  
import logging  
from PIL import Image  
from pathlib import Path  

# 添加路径以便导入自定义数据集加载器  
sys.path.append('/home/nw/Codes/DatasetLoader')  
from WHURS19_DatasetLoader import WHURS19DatasetLoader  

# 新的标签列表  
LABELS_16_CLASSES = [  
    "arable land", "grassland", "woodland", "commercial area", "factory area",  
    "mining area", "power station", "sports land", "detached house",  
    "airport area", "highway area", "port area",  
    "railway area", "bare land", "lake", "river"  
] 

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

class RemoteCLIPZeroShotClassifier:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  
        
        # 加载CLIP模型和预处理函数  
        self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        self.tokenizer = open_clip.get_tokenizer(self.model_name)  
        
        # 加载模型检查点  
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))  
        self.model = self.model.to(self.device).eval()  

        self.label_texts = LABELS_16_CLASSES  
        self.label_text_features = self._get_text_features(self.label_texts).to(torch.float32)  

    def _get_text_features(self, texts):  
        tokenized_texts = self.tokenizer(texts).to(self.device)  
        with torch.no_grad():  
            text_features = self.model.encode_text(tokenized_texts)  
        return text_features / text_features.norm(dim=-1, keepdim=True)  

    def get_image_features(self, image):  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(image)  
        return image_features / image_features.norm(dim=-1, keepdim=True)  

    def classify_image(self, query_image):  
        query_image = query_image.unsqueeze(0).to(self.device)  
        image_features = self.get_image_features(query_image).to(torch.float32)  
        with torch.no_grad():  
            similarities = (100.0 * image_features @ self.label_text_features.T).softmax(dim=-1)  
            top_probs, top_labels = similarities.cpu().topk(1, dim=-1)  
            label = top_labels.item()
            prob = top_probs.item()
        return self.label_texts[label], prob # 返回概率和标签  

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

        predicted_label, probability = classifier.classify_image(query_image)  
        logging.info(f"File: {file_name}\n"  
                     f"Path: {image_path}\n"  
                     f"Predicted label: {predicted_label}\n"  
                     f"Probability: {probability:.2%}\n"  
                     f"{'-'*40}")  

def evaluate_classifier(classifier, dataset_loader):  
    total_images = 0  
    correct_predictions = 0  

    for image, ground_truth_label, image_path in dataset_loader:  
        ground_truth_label = LABELS_16_CLASSES[ground_truth_label]  # 使用新的标签索引  

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

    classifier = RemoteCLIPZeroShotClassifier(ckpt_path, model_name)  

    #调用分类函数进行分类  
    query_folder_path = '/home/nw/Codes/RemoteCLIP/assets/GF2_Selected_50'  
    classify_images_in_folder(query_folder_path, classifier)  

    # # 使用 WHURS19 数据集加载器  
    # whurs19_folder_path = '/mnt/d/nw/Datasets/Classification-12/WHU-RS19'  # WHURS19 数据集的路径  
    # whurs19_dataset_loader = WHURS19DatasetLoader(  
    #     data_path=whurs19_folder_path,  
    #     preprocess_func=classifier.preprocess_func  
    # )  

    # # 调用评估函数计算在 WHURS19 数据集上的精度  
    # evaluate_classifier(classifier, whurs19_dataset_loader)