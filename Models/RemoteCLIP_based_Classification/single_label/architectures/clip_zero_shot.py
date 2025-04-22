import torch
from ..core.base import BaseCLIPClassifier

class ZeroShotClassifier(BaseCLIPClassifier):  
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.text_features = None  
        self._prepare_text_features()  

    def _prepare_text_features(self):  
        if not self.labels:  
            raise ValueError("必须提供标签列表")  
        self.text_features = self._get_text_features(self.labels).to(torch.float32)  

    def train(self, train_loader=None, val_loader=None, **kwargs):  
        self._prepare_text_features()  
        if val_loader is not None:  
            acc = self.evaluate(val_loader)  
            self.logger.info(f'Validation accuracy: {acc['accuracy']:.4f}')  

    def evaluate(self, data_loader):  
        correct, total = 0, 0  
        for img_batch, label_batch in data_loader:  
            img_features = self._get_image_features(img_batch)  
            similarity = (100.0 * img_features @ self.text_features.T).softmax(dim=-1)  
            pred_indices = similarity.argmax(dim=-1).cpu().numpy()  
            pred_labels = [self.labels[i] for i in pred_indices]  
            for pred, gt in zip(pred_labels, label_batch):  
                if pred == gt:  
                    correct += 1  
            total += len(label_batch)  
        return {'accuracy': correct / total if total else 0}  

    def _predict_single(self, img_path_or_image):  
        if not self.labels:  
            raise RuntimeError("未设置分类标签")  
        image = self._load_image(img_path_or_image) if isinstance(img_path_or_image, str) else img_path_or_image  
        img_tensor = self.preprocess_func(image).unsqueeze(0)  
        img_features = self._get_image_features(img_tensor)  
        similarity = (100.0 * img_features @ self.text_features.T).softmax(dim=-1)  
        probs, indices = similarity.topk(3)  
        return {  
            'label': self.labels[indices[0][0].item()],  
            'top3': [(self.labels[i.item()], float(p)) for p, i in zip(probs[0], indices[0])]  
        }  