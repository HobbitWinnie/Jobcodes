import torch
from nw.Codes.Models.RemoteCLIP_based_Classification.single_label.core.base import BaseCLIPClassifier  

class FewShot(BaseCLIPClassifier):  
    def __init__(self, *args, num_shots=5, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.support_set = {'images': [], 'labels': []}  
        self.num_shots = num_shots  

    def load_support_set(self, dataset):  
        """加载支持集"""  
        class_counts = {cls: 0 for cls in dataset.classes}  
        
        for item in dataset:  
            if all(c >= self.num_shots for c in class_counts.values()):  
                break  
                
            if class_counts[item['label']] < self.num_shots:  
                self.support_set['images'].append(item['image'])  
                self.support_set['labels'].append(item['label'])  
                class_counts[item['label']] += 1  

    def classify_image(self, image):  
        """实现少样本分类"""  
        if not self.support_set['images']:  
            raise RuntimeError("Support set not loaded")  
            
        # 预处理查询图像  
        query_img = self.preprocess_func(image).unsqueeze(0)  
        query_feat = self._get_image_features(query_img)  
        
        # 支持集特征  
        support_images = torch.stack([  
            self.preprocess_func(img) for img in self.support_set['images']  
        ])  
        support_feats = self._get_image_features(support_images)  
        text_feats = self._get_text_features(self.support_set['labels'])  
        
        # 计算相似度  
        image_sim = (query_feat @ support_feats.T) * 100  
        text_sim = (query_feat @ text_feats.T) * 100  
        combined = (image_sim + text_sim).softmax(dim=-1)  
        
        # 获取Top3结果  
        probs, indices = combined.topk(3)  
        return [  
            (self.support_set['labels'][i], p.item())   
            for p, i in zip(probs[0], indices[0])  
        ]  