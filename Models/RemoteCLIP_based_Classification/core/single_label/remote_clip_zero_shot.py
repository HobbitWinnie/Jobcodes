import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

import torch  
import open_clip  
from PIL import Image  
import pandas as pd  


class RemoteCLIPZeroShotClassifier:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', labels=None, device=None):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  

        # Load CLIP model and preprocessing function  
        self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        self.tokenizer = open_clip.get_tokenizer(self.model_name)  

        # Load model checkpoint  
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))  
        self.model = self.model.to(self.device).eval()  

        self.label_texts = labels if labels else []  
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
        query_image = self.preprocess_func(query_image).unsqueeze(0).to(self.device)  
        image_features = self.get_image_features(query_image).to(torch.float32)  
        with torch.no_grad():  
            similarities = (100.0 * image_features @ self.label_text_features.T).softmax(dim=-1)  
            top_probs, top_labels = similarities.cpu().topk(3, dim=-1)  
            
            top_labels = top_labels.squeeze().tolist()  
            top_probs = top_probs.squeeze().tolist()  
            top_labels = [self.label_texts[idx] for idx in top_labels]  

        return top_labels, top_probs  # 返回前3个标签字符串和概率  

    def classify_images_in_folder(self, folder_path, output_csv):  
        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                image = Image.open(img_path).convert("RGB")  
                # image = self.preprocess_func(image).unsqueeze(0)  
                top_labels, top_probs = self.classify_image(image)  
                result = {"filename": img_name}  
                for i in range(3):  
                    result[f"top{i+1}_label"] = top_labels[i]  
                    result[f"top{i+1}_prob"] = top_probs[i]  
                results.append(result)  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  
        print(f"Results saved to `{output_csv}`")  
