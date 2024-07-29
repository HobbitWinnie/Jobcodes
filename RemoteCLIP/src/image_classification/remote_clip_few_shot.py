import torch  
import open_clip  

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
        query_image: A single image (pil) to classify.  
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

            # Get top prediction  
            top_probs, top_labels = combined_similarities.cpu().topk(1, dim=-1)  
            label_index = top_labels.item()  
            prob = top_probs.item()  
            label = support_labels[label_index]  

        return label, prob  