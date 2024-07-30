import torch  
import numpy as np  
from sklearn.svm import LinearSVC  
from sklearn.preprocessing import MultiLabelBinarizer  
from sklearn.utils import shuffle  
import open_clip  
from PIL import Image  
import pandas as pd  
import os  
from sklearn.preprocessing import StandardScaler  

class RemoteCLIPClassifierRankSVM:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  

        # Load the CLIP model and preprocessing function  
        self.model, _, self.preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        self.tokenizer = open_clip.get_tokenizer(self.model_name)  

        # Load model checkpoint  
        ckpt = torch.load(ckpt_path, map_location='cpu')  
        self.model.load_state_dict(ckpt)  
        self.model = self.model.to(self.device).eval()  

        self.rank_svm = None  
        self.mlb = MultiLabelBinarizer()  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy()  

    def fit_rank_svm(self, dataloader, C=1.0):  
        train_image_features = []  
        train_labels = []  
        
        for images, labels, _ in dataloader:  
            features = self.get_image_features(images)  
            train_image_features.append(features)  
            train_labels.extend(labels)  
        
        train_image_features = np.vstack(train_image_features)  
        train_labels = np.array(train_labels)  

        # Multi-label Binarization  
        self.mlb.fit(train_labels)  
        binarized_labels = self.mlb.transform(train_labels)  
        
        pairs_train, pairs_label = self._create_pairs(train_image_features, binarized_labels)  

        scaler = StandardScaler()  
        pairs_train = pairs_train.astype(np.float32)  # 确保 32-bit 浮点  
        pairs_train = scaler.fit_transform(pairs_train)  

        pairs_label = pairs_label.astype(np.int8)  # 确保标签为 8-bit 整数  

        self.rank_svm = LinearSVC(C=C)  
        self.rank_svm.fit(pairs_train, pairs_label)

    def _create_pairs(self, X, y):  
        """  
        Create pairs for rank SVM.  
        """  
        pairs = []  
        targets = []  
        
        num_samples, num_labels = y.shape  
        
        for i in range(num_samples):  
            for j in range(num_samples):  
                if i != j:  
                    diff = X[i] - X[j]  
                    label_diff = y[i] - y[j]  
                    for k in range(len(label_diff)):  
                        if label_diff[k] != 0:  
                            pairs.append(diff.astype(np.float32))  # 确保转换为 32-bit 浮点  
                            targets.append(np.sign(label_diff[k]))  

        pairs, targets = shuffle(pairs, targets)  
        return np.array(pairs, dtype=np.float32), np.array(targets, dtype=np.int8)  # 确保对和标签类型

    def classify_image(self, query_image):  
        query_image = query_image.to(self.device)  
        query_image_features = self.get_image_features(query_image.unsqueeze(0))  
        
        # For multiple labels prediction  
        probas = self.rank_svm.decision_function(query_image_features)  
        sorted_indices = np.argsort(probas.flatten())[::-1]  
        ranked_labels = self.mlb.classes_[sorted_indices]  
        
        return ranked_labels  
    
    def classify_images_in_folder(self, folder_path, output_csv):  
        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                image = Image.open(img_path).convert("RGB")  
                image = self.preprocess_func(image).unsqueeze(0)  
                labels = self.classify_image(image)  
                results.append({"filename": img_name, "labels": labels})  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  
        print(f"Results saved to `{output_csv}`")  