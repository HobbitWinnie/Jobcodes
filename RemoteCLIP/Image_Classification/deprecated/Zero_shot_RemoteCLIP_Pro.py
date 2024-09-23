import os  
import torch  
import open_clip  
import logging  
from PIL import Image  
from pathlib import Path  
import shutil  
from concurrent.futures import ThreadPoolExecutor, as_completed  

# 新的标签列表
# LABELS_16_CLASSES = [  
#     "林地", "草地", "耕地", "商业区",  
#     "工业区", "矿区", "发电站", "住宅区",  
#     "机场", "道路", "港口区", "桥梁",  
#     "火车站", "高架桥", "裸地", "开阔水域", "小水体"  
# ]  

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
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  

        # 尝试加载CLIP模型和预处理函数  
        try:  
            self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(model_name)  
            self.tokenizer = open_clip.get_tokenizer(model_name)  
            # 加载模型检查点  
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))  
            self.model = self.model.to(self.device).eval()  
        except Exception as e:  
            logging.error(f"加载模型失败: {e}")  
            raise e  

        # 文本特征准备  
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
        return self.label_texts[label], prob  # 返回概率和标签  

def organize_images(classified_images, output_path, labels):  
    """为分类结果中的每个标签生成文件夹，并将图像移动到对应的标签文件夹。"""  

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

def process_image(image_file, classifier):  
    try:  
        with Image.open(image_file) as image:  
            query_image = image.convert('RGB')  
            query_image = classifier.preprocess_func(query_image)  
        predicted_label, _ = classifier.classify_image(query_image)  
        return image_file, predicted_label  
    except Exception as e:  
        logging.warning(f"处理图像 {image_file} 失败: {e}")  
        return image_file, None  

def classify_images_in_folder(folder_path, classifier, labels, output_path=None, max_workers=8):  
    classified_images = []  

    if not os.path.exists(folder_path):  
        logging.error(f"文件夹路径 {folder_path} 不存在。")  
        return  

    image_files = [f for f in Path(folder_path).rglob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]  

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
        futures = {executor.submit(process_image, image_file, classifier): image_file for image_file in image_files}  
        for future in as_completed(futures):  
            image_file = futures[future]  
            try:  
                result = future.result()  
                if result[1] is not None:  
                    classified_images.append(result)  
            except Exception as e:  
                logging.error(f"处理图像 {image_file} 时出错: {e}")  

    if output_path:  
        organize_images(classified_images, output_path=output_path, labels=labels)  

if __name__ == "__main__":  
    # 更新后的路径  
    ckpt_path = '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-L-14.pt'  
    query_folder_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_512_dataset/image'  
    output_folder_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_512_dataset'

    try:  
        # 初始化分类器  
        classifier = RemoteCLIPZeroShotClassifier(ckpt_path)  
        
        # 分类图像  
        classify_images_in_folder(query_folder_path, classifier, LABELS_16_CLASSES, output_path=output_folder_path)  
    except Exception as e:  
        logging.error(f"分类过程中发生错误: {e}")