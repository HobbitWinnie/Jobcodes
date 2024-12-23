import os  
import torch  
import open_clip  
import logging  
from PIL import Image  
from pathlib import Path  
from concurrent.futures import ThreadPoolExecutor, as_completed  
import csv  

# 标签列表  
LABELS_16_CLASSES = [  
    "Forest", "Grassland", "Farmland", "Commercial Area",  
    "Industrial Area", "Mining Area", "Power Station", "Residential Area",  
    "Airport", "Road", "Port Area", "Bridge",  
    "Train Station", "Viaduct", "Bare Land", "Open Water", "Small Water Body"  
]  

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

class RemoteCLIPZeroShotClassifier:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  

        try:  
            self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(model_name)  
            self.tokenizer = open_clip.get_tokenizer(model_name)  
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))  
            self.model = self.model.to(self.device).eval()  
        except Exception as e:  
            logging.error(f"加载模型失败: {e}")  
            raise e  

        self.label_text_features = self._get_text_features(LABELS_16_CLASSES).to(torch.float32)  

    def _get_text_features(self, texts):  
        tokenized_texts = self.tokenizer(texts).to(self.device)  
        with torch.no_grad():  
            text_features = self.model.encode_text(tokenized_texts)  
        return text_features / text_features.norm(dim=-1, keepdim=True)  

    def classify_image(self, image):  
        image = image.unsqueeze(0).to(self.device)  
        with torch.no_grad():  
            image_features = self.model.encode_image(image)  
            image_features /= image_features.norm(dim=-1, keepdim=True)  
            similarities = (100.0 * image_features @ self.label_text_features.T).softmax(dim=-1)  
            threshold = 0.5  
            return [(LABELS_16_CLASSES[idx], prob) for idx, prob in enumerate(similarities[0]) if prob >= threshold]  

def process_image(image_file, classifier):  
    try:  
        with Image.open(image_file) as img:  
            img = img.convert('RGB')  
            img = classifier.preprocess_func(img)  
        return image_file, classifier.classify_image(img)  
    except Exception as e:  
        logging.warning(f"处理图像 {image_file} 失败: {e}")  
        return image_file, None  

def classify_images_in_folder(folder_path, classifier, csv_output, max_workers=8):  
    if not os.path.exists(folder_path):  
        logging.error(f"文件夹路径 {folder_path} 不存在。")  
        return  

    image_files = list(Path(folder_path).rglob('*.[jp][pn]g'))  
    classified_images = []  

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
        futures = {executor.submit(process_image, image_file, classifier): image_file for image_file in image_files}  
        for future in as_completed(futures):  
            image_file, predicted_labels = future.result()  
            if predicted_labels:  
                binary_vector = [1 if label in (label for label, _ in predicted_labels) else 0 for label in LABELS_16_CLASSES]  
                classified_images.append((os.path.basename(image_file), binary_vector))  

    with open(csv_output, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)  
        writer.writerow(['image'] + LABELS_16_CLASSES)  
        for image_file, binary_vector in classified_images:  
            writer.writerow([image_file] + binary_vector)  

if __name__ == "__main__":  
    # ckpt_path = '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-L-14.pt'  
    ckpt_path = '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-RN50.pt'  

    query_folder_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_512_dataset/Images'  
    labels_output_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_512_dataset/Labels_RN50'  

    os.makedirs(labels_output_path, exist_ok=True)  

    try:  
        model_name = 'RN50'     #'RN50' or 'ViT-B-32' or 'ViT-L-14'
        classifier = RemoteCLIPZeroShotClassifier(ckpt_path, model_name)  
        
        for subfolder in Path(query_folder_path).iterdir():  
            if subfolder.is_dir():  
                csv_output_path = os.path.join(labels_output_path, f"{subfolder.name}.csv")  
                classify_images_in_folder(subfolder, classifier, csv_output=csv_output_path)  
    except Exception as e:  
        logging.error(f"分类过程中发生错误: {e}")