import torch  
import numpy as np  
from sklearn.svm import LinearSVC  
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.utils import shuffle  
import open_clip   
from PIL import Image  
import pandas as pd  
import os  


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
        with torch.no_grad():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().numpy().astype(np.float32)  # Ensure 32-bit float  

    def fit_rank_svm(self, dataloader, C=1.0):  
        train_image_features = []  
        train_labels = []  
        
        for images, labels, _ in dataloader:  
            image_features = self.get_image_features(images)  
            train_image_features.append(image_features)  
            train_labels.extend(labels.numpy())  

        train_image_features = np.vstack(train_image_features)  
        train_labels = np.array(train_labels)  

        # 将每个标签向量转换为标签集合列表  
        train_labels_list = []  
        for row in train_labels:  
            train_labels_list.append(np.where(row == 1)[0].tolist())  

        # Multi-label Binarization  
        binarized_labels = self.mlb.fit_transform(train_labels_list)  
        
        pairs_train, pairs_label = self._create_pairs(train_image_features, binarized_labels)  

        # Scale the feature pairs  
        scaler = StandardScaler()  
        pairs_train = scaler.fit_transform(pairs_train.astype(np.float32))  # Ensure 32-bit float and scale  

        pairs_label = pairs_label.astype(np.int32)  # Ensure 32-bit int for labels  
        print(f"Pairs train dtype: {pairs_train.dtype}, label dtype: {pairs_label.dtype}")  

        self.rank_svm = LinearSVC(C=C)  
        self.rank_svm.fit(pairs_train, pairs_label)  

    def _create_pairs(self, X, y, max_pairs=100000):  
        """  
        Create pairs for rank SVM.  
        """  
        pairs = []  
        targets = []  
        num_samples = y.shape[0]  
        
        # 使用随机选择的方法来减少内存使用  
        sampled_indices = np.random.choice(num_samples, size=min(max_pairs, num_samples), replace=False)  
        pairs_count = 0  

        for i in sampled_indices:  
            for j in np.random.choice(num_samples, size=min(max_pairs, num_samples), replace=False):  
                if i != j:  
                    diff = X[i] - X[j]  
                    label_diff = y[i] - y[j]  
                    for k in range(len(label_diff)):  
                        if label_diff[k] != 0:  
                            pairs.append(diff)  
                            targets.append(int(np.sign(label_diff[k])))  
                            pairs_count += 1  
                            if pairs_count >= max_pairs:  
                                break  
                if pairs_count >= max_pairs:  
                    break  
            if pairs_count >= max_pairs:  
                break  

        print("Max diff: ", np.max(pairs))  
        print("Min diff: ", np.min(pairs))   

        pairs, targets = shuffle(pairs, targets)  
        return np.array(pairs), np.array(targets)  


    def classify_image(self, query_image, top_k=3):  
        query_image = query_image.to(self.device)  
        query_image_features = self.get_image_features(query_image)
        scores = self.rank_svm.decision_function(query_image_features)  
        
        top_indices = np.argsort(scores[0])[::-1][:top_k]  
        top_labels = self.mlb.classes_[top_indices]  
        top_scores = scores[0][top_indices]  

        return top_labels, top_scores 

    
    def classify_images_in_folder(self, folder_path, output_csv):  
        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                image = Image.open(img_path).convert("RGB")  
                image = self.preprocess_func(image).unsqueeze(0)  
                top_labels, top_scores = self.classify_image(self.model, image, self.tokenizer, self.device, self.rank_svm, self.mlb, top_k=3)  
                results.append({"filename": img_name, "top_labels": top_labels, "top_scores": top_scores})  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  
        print(f"Results saved to `{output_csv}`")