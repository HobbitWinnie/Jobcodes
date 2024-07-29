import torch  
import open_clip  
from PIL import Image  
import pandas as pd  
import os  

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

    def few_shot_classify(self, support_images, support_labels, query_image):  
        """  
        support_images: List of PIL Images, few-shot examples.  
        support_labels: List of corresponding labels for the support_images.  
        query_image: A single image (PIL) to classify.  
        """  

        # Feature extraction for the support set  
        support_image_features = self._get_image_features(support_images).to(torch.float32)  
        support_text_features = self._get_text_features(support_labels).to(torch.float32)  

        # Feature extraction for the query image  
        query_image = self.preprocess_func(query_image).unsqueeze(0).to(self.device)  
        query_image_features = self._get_image_features([query_image]).to(torch.float32)  

        # Calculate similarities between query image and support set  
        with torch.no_grad():  
            image_similarities = 100.0 * query_image_features @ support_image_features.T  
            text_similarities = 100.0 * query_image_features @ support_text_features.T  

            # Combine similarities (simple addition)  
            combined_similarities = (image_similarities + text_similarities).softmax(dim=-1)  

            # Get top 3 predictions  
            top_probs, top_labels = combined_similarities.cpu().topk(3, dim=-1)  

            # Extract top 3 labels and their corresponding probabilities  
            results = []  
            for idx in range(top_probs.size(1)):  
                label_index = top_labels[0, idx].item()  
                prob = top_probs[0, idx].item()  
                label = support_labels[label_index]  
                results.append((label, prob))  

        return results  

    def classify_images_in_folder(self, support_images, support_labels, folder_path, output_csv):  
        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                query_image = Image.open(img_path).convert("RGB")  
                top_3_labels = self.few_shot_classify(support_images, support_labels, query_image)  
                results.append({  
                    "filename": img_name,  
                    "top_1_label": top_3_labels[0][0],  
                    "top_1_prob": top_3_labels[0][1],  
                    "top_2_label": top_3_labels[1][0],  
                    "top_2_prob": top_3_labels[1][1],  
                    "top_3_label": top_3_labels[2][0],  
                    "top_3_prob": top_3_labels[2][1]  
                })  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  

# 示例用法  
# checkpoint_path = "path/to/checkpoint"  
# model_name = "ViT-L-14"  
# classifier = RemoteCLIPFewShotClassifier(ckpt_path=checkpoint_path, model_name=model_name)  

# support_images = [Image.open(path) for path in support_image_paths]  # 假设有一组支持图像路径  
# support_labels = ["label1", "label2", "label3"]  # 对应的支持标签  
# classifier.classify_images_in_folder(support_images, support_labels, 'path/to/images_folder', 'output.csv')