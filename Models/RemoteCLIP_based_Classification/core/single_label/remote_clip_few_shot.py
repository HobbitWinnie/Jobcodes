import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

import torch  
import open_clip  
from PIL import Image  
import pandas as pd  
from torch.utils.data import Dataset  


class RemoteCLIPFewShotClassifier:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  

        # Load CLIP model and preprocessing function  
        self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        self.tokenizer = open_clip.get_tokenizer(self.model_name)  

        # Load model checkpoint  
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))  
        self.model = self.model.to(self.device).eval()  
        
        self.support_images = []  
        self.support_labels = []  

    def _get_text_features(self, texts):  
        tokenized_texts = self.tokenizer(texts).to(self.device)  
        with torch.no_grad():  
            text_features = self.model.encode_text(tokenized_texts)  
        return text_features / text_features.norm(dim=-1, keepdim=True)  

    def _get_image_features(self, images):  
        images = torch.stack([self.preprocess_func(image).to(self.device) for image in images])  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(images)  
        return image_features / image_features.norm(dim=-1, keepdim=True)  
    
    def load_support_dataset(self, dataset, num_shots=5):  
        class_sample_count = {cls: 0 for cls in dataset.classes}  
        for _, label, image, _ in dataset:  
            if class_sample_count[label] < num_shots:  
                self.support_images.append(image)  # 保存原始的 PIL 图像  
                self.support_labels.append(label)  
                class_sample_count[label] += 1  
            if all(count >= num_shots for count in class_sample_count.values()):  
                break  

    def few_shot_classify(self, query_image):  
        """  
        Classify a query image based on the support images and labels using few-shot learning.  
        """  
        # Feature extraction for the support set  
        support_image_features = self._get_image_features(self.support_images).to(torch.float32)  
        support_text_features = self._get_text_features(self.support_labels).to(torch.float32)  
        
        # Feature extraction for the query image  
        query_image = self.preprocess_func(query_image).unsqueeze(0).to(self.device)  # 预处理并转为 tensor  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            query_image_features = self.model.encode_image(query_image).to(torch.float32)  

        # Calculate similarities between query image and support set  
        with torch.no_grad():  
            image_similarities = 100.0 * query_image_features @ support_image_features.T  
            text_similarities = 100.0 * query_image_features @ support_text_features.T  

            # Combine similarities (simple addition)  
            combined_similarities = (image_similarities + text_similarities).softmax(dim=-1)  

            # Get top 3 predictions  
            top_probs, top_labels = combined_similarities.cpu().topk(3, dim=-1)  
            top_labels = top_labels.squeeze().tolist()  
            top_probs = top_probs.squeeze().tolist()  
            top_labels = [self.support_labels[idx] for idx in top_labels]  

        return top_labels, top_probs  

    def classify_images_in_folder(self, folder_path, output_csv):  
        if not os.path.exists(folder_path):  
            print(f"Error: Folder `{folder_path}` does not exist.")  
            return  

        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                query_image = Image.open(img_path).convert("RGB")  # 确保是 PIL 图像  
                top_labels, top_probs = self.few_shot_classify(query_image)  
                results.append({  
                    "filename": img_name,  
                    "top1_label": top_labels[0], "top1_prob": top_probs[0],  
                    "top2_label": top_labels[1], "top2_prob": top_probs[1],  
                    "top3_label": top_labels[2], "top3_prob": top_probs[2]  
                })  

        if results:  
            df = pd.DataFrame(results)  
            df.to_csv(output_csv, index=False)  
            print(f"Results saved to `{output_csv}`")  
        else:  
            print("No valid images found to classify.")