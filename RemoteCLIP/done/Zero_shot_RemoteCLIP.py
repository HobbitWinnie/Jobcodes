import os  
from PIL import Image  
from pathlib import Path  
import torch  
import open_clip  
import logging  


LABEL_MAPPING = {  
    0: 'Airport',  
    1: 'Beach',  
    2: 'Bridge',  
    3: 'Commercial',  
    4: 'Desert',  
    5: 'Farmland',  
    6: 'Forest',  
    7: 'Industrial',  
    8: 'Meadow',  
    9: 'Mountain',  
    10: 'Park',  
    11: 'Parking',  
    12: 'Pond',  
    13: 'Port',  
    14: 'Residential',  
    15: 'River',  
    16: 'Viaduct',  
    17: 'footballField',  
    18: 'railwayStation'  
}  

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

class RemoteCLIPZeroShotClassifier:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  
        
        # Load CLIP model and preprocessing function  
        self.model, self.preprocess_func, _ = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)  
        
        # Load model checkpoint  
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))  
        self.model = self.model.to(self.device).eval()  

        self.label_texts = [LABEL_MAPPING[i] for i in range(len(LABEL_MAPPING))]  
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
        return LABEL_MAPPING[top_labels.item()], top_probs.item()  # 返回概率和标签  

def classify_images_in_folder(folder_path, classifier):  
    if not os.path.exists(folder_path):  
        print(f"Folder path {folder_path} does not exist.")  
        return  

    for image_path in Path(folder_path).rglob('*.jpg'):  
        file_name = os.path.basename(image_path)  
        query_image = Image.open(image_path).convert('RGB')  
        query_image = classifier.preprocess_func(query_image)  

        predicted_label, probability = classifier.classify_image(query_image)  
        logging.info(f"File: {file_name}\n"  
                     f"Path: {image_path}\n"  
                     f"Predicted label: {predicted_label}\n"  
                     f"Probability: {probability:.2%}\n"  
                     f"{'-'*40}") 
if __name__ == "__main__":  
    query_folder_path = '/mnt/d/nw/Datasets/Classification-12/testdata'  
    ckpt_path = '/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-ViT-L-14.pt'
    model_name='ViT-L-14'

    classifier = RemoteCLIPZeroShotClassifier(ckpt_path, model_name)  
    classify_images_in_folder(query_folder_path, classifier)