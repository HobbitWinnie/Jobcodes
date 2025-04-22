import torch
from ..core.base import BaseCLIPClassifier

class FewShotClassifier(BaseCLIPClassifier):  
    def __init__(self, *args, num_shots=5, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.num_shots = num_shots  
        self.support_set = {'images': [], 'labels': []}  
        self.classes = []  

    def train(self, dataset, **kwargs):  
        """dataset: 可迭代，元素为 {'image': PIL.Image, 'label': label}"""  
        class_counts = {}  
        self.support_set = {'images': [], 'labels': []}  
        for item in dataset:  
            label = item['label']  
            if class_counts.get(label, 0) < self.num_shots:  
                self.support_set['images'].append(item['image'])  
                self.support_set['labels'].append(label)  
                class_counts[label] = class_counts.get(label, 0) + 1  
        self.classes = sorted(set(self.support_set['labels']))  

    def evaluate(self, data_loader):  
        correct, total = 0, 0  
        for batch in data_loader:  
            for item in batch:  
                pred = self._predict_single(item['image'])  
                if pred['label'] == item['label']:  
                    correct += 1  
                total += 1  
        return {'accuracy': correct / total if total else 0}  

    def _predict_single(self, img_path_or_image):  
        if not self.support_set['images']:  
            raise RuntimeError("Support set not loaded")  
        # 自动适应路径或已加载图片  
        image = self._load_image(img_path_or_image) if isinstance(img_path_or_image, str) else img_path_or_image  
        query_img = self.preprocess_func(image).unsqueeze(0)  
        query_feat = self._get_image_features(query_img)  

        support_images = torch.stack([self.preprocess_func(img) for img in self.support_set['images']])  
        support_feats = self._get_image_features(support_images)  
        text_feats = self._get_text_features(self.support_set['labels'])  

        image_sim = (query_feat @ support_feats.T) * 100  
        text_sim = (query_feat @ text_feats.T) * 100  
        combined = (image_sim + text_sim).softmax(dim=-1)  
        probs, indices = combined.topk(3)  
        pred_label = self.support_set['labels'][indices[0][0].item()]  
        return {  
            'label': pred_label,  
            'top3': [  
                (self.support_set['labels'][i.item()], float(p))  
                for p, i in zip(probs[0], indices[0])  
            ]  
        }  