import torch
from ..core.base import BaseCLIPClassifier

class FewShotClassifier(BaseCLIPClassifier):  
    def __init__(self, *args, num_shots=5, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.num_shots = num_shots  
        self.support_set = {'images': [], 'labels': []}  
        self.classes = []  

    def train(self, support_loader, val_loader):  
        """  
        data_loader: batch 结构为 (img_batch, label_batch, ...)  
        取每个batch的label_batch，逐标签收集num_shots数量的图片  
        """          
        class_counts = {}  
        self.support_set = {'images': [], 'labels': []}  
        for img_batch, label_batch, *_ in support_loader:  
            for img, label in zip(img_batch, label_batch):  
                if class_counts.get(label, 0) < self.num_shots:  
                    # img可能是Tensor或PIL.Image，视自己的预处理方式选择  
                    self.support_set['images'].append(img)  
                    self.support_set['labels'].append(label)  
                    class_counts[label] = class_counts.get(label, 0) + 1  
        self.classes = sorted(set(self.support_set['labels']))  

        acc = self.evaluate(val_loader)  
        self.logger.info(f"Validation accuracy: {acc['accuracy']:.4f}")  

    def evaluate(self, data_loader):  
        correct, total = 0, 0  
        support_images_tensor = torch.stack(self.support_set['images']).to(self.main_device)  
        support_feats = self._get_image_features(support_images_tensor)  
        text_feats = self._get_text_features(self.support_set['labels']).float()  
        support_feats = support_feats.float()  
        support_labels = self.support_set['labels']  

        for img_batch, label_batch, *_ in data_loader:  
            img_batch = img_batch.to(self.main_device)  
            query_feats = self._get_image_features(img_batch).float()  
            image_sim = (query_feats @ support_feats.T) * 100  
            text_sim = (query_feats @ text_feats.T) * 100  
            combined = (image_sim + text_sim).softmax(dim=-1)  
            pred_indices = combined.argmax(dim=-1).cpu().numpy()  

            for idx, gt in zip(pred_indices, label_batch):  
                pred_label = support_labels[idx]  
                if str(pred_label) == str(gt):  
                    correct += 1  
                total += 1  

        return {'accuracy': correct / total if total else 0}  
        

    def _predict_single(self, img_path_or_image):  
        if not self.support_set['images']:  
            raise RuntimeError("Support set not loaded")  
            
        # 只对预测目标本身预处理  
        if isinstance(img_path_or_image, str):  
            image = self._load_image(img_path_or_image)  
            image_tensor = self.preprocess_func(image).unsqueeze(0).to(self.main_device)  
        else:  
            # 如果传入就是已经处理好的 tensor  
            image_tensor = img_path_or_image.unsqueeze(0).to(self.main_device)  
            
        query_feat = self._get_image_features(image_tensor)  
        
        # 支持集图片已为 tensor，直接堆叠  
        support_images = torch.stack(self.support_set['images']).to(self.main_device)  
        support_feats = self._get_image_features(support_images)  
        text_feats = self._get_text_features(self.support_set['labels'])  
        
        # 保证特征 dtype 一致（推荐全部 float32）  
        query_feat = query_feat.float()  
        support_feats = support_feats.float()  
        text_feats = text_feats.float()  
        
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