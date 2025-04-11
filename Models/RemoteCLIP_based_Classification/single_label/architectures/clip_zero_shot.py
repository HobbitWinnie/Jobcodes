import torch  
from core.base import BaseCLIPClassifier  

class ZeroShot(BaseCLIPClassifier):  
    def __init__(self, *args, labels=None, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.labels = labels or []  
        self._prepare_text_features()  

    def _prepare_text_features(self):  
        """预计算文本特征"""  
        if not self.labels:  
            raise ValueError("必须提供标签列表")  
            
        self.text_features = self._get_text_features(self.labels)  
        self.text_features = self.text_features.to(torch.float32)  

    def classify_image(self, image):  
        """零样本分类"""  
        if not self.labels:  
            raise RuntimeError("未设置分类标签")  
            
        # 预处理和特征提取  
        img_tensor = self.preprocess_func(image).unsqueeze(0)  
        img_features = self._get_image_features(img_tensor)  
        
        # 计算相似度  
        similarity = (100.0 * img_features @ self.text_features.T).softmax(dim=-1)  
        probs, indices = similarity.topk(3)  
        
        return [  
            (self.labels[i.item()], p.item())   
            for p, i in zip(probs[0], indices[0])  
        ]  