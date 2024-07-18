import torch  
import open_clip  


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
        query_image = query_image.unsqueeze(0).to(self.device)  
        image_features = self.get_image_features(query_image).to(torch.float32)  
        with torch.no_grad():  
            similarities = (100.0 * image_features @ self.label_text_features.T).softmax(dim=-1)  
            top_probs, top_labels = similarities.cpu().topk(1, dim=-1)  
        #     label = top_labels.item()  
        #     prob = top_probs.item()  
        # return self.label_texts[label], prob  # 返回概率和标签
            label_index = top_labels.item()  
            prob = top_probs.item()  
            label = self.label_texts[label_index]  
        return label, prob  # 返回标签字符串和概率  