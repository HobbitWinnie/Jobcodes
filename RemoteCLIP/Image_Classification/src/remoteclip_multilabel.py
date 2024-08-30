import os  
import sys  
import torch  
from torch import nn, optim  
import numpy as np  
import pandas as pd  
import open_clip  
from PIL import Image  
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score  
from torch.optim.lr_scheduler import StepLR  

sys.path.append('/home/nw/Codes/RemoteCLIP/src/loss_functions')  
from loss_functions import FocalLoss, DiceLoss, LabelSmoothingLoss  # 导入损失函数  


class MultiLabelClassifier:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.model_name = model_name  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  
        print(f"Running on device: {self.device}")  

        self.model, _, preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        self.preprocess_func = preprocess_func  
        self.tokenizer = open_clip.get_tokenizer(self.model_name)  

        ckpt = torch.load(ckpt_path, map_location='cpu')  
        self.model.load_state_dict(ckpt)  
        self.model = self.model.to(self.device).eval()  

        self.fc = None  
        self.optimizer = None  # 初始化优化器  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features  

    def train_model(self, dataloader, num_labels, num_epochs=2, lr=1e-4, loss_type='bce', **kwargs):  
        output_dim = self.model.module.visual.output_dim if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.visual.output_dim  

        # 添加新的全连接层  
        self.fc = nn.Sequential(  
            nn.Linear(output_dim, 512),  
            nn.ReLU(),  
            nn.Linear(512, 256),  
            nn.ReLU(),  
            nn.Linear(256, num_labels)  
        ).to(self.device)  

        if loss_type == 'bce':  
            criterion = nn.BCEWithLogitsLoss()  
        elif loss_type == 'focal':  
            criterion = FocalLoss(**kwargs)  
        elif loss_type == 'dice':  
            criterion = DiceLoss()  
        elif loss_type == 'label_smoothing':  
            criterion = LabelSmoothingLoss(classes=num_labels)  
        else:  
            raise ValueError("Unsupported loss type. Choose from 'bce', 'focal', 'dice', 'label_smoothing'.")  
        
        self.optimizer = optim.AdamW(self.fc.parameters(), lr=lr)  
        scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)  
        scaler = torch.cuda.amp.GradScaler()  
        
        self.fc.train()  
        for epoch in range(num_epochs):  
            total_loss = 0.0  
            for i, (images, targets, _) in enumerate(dataloader):  
                images = images.to(self.device)  
                targets = targets.to(self.device).float()  
                
                self.optimizer.zero_grad()  
                
                with torch.cuda.amp.autocast():  
                    features = self.get_image_features(images)  
                    outputs = self.fc(features)  
                    loss = criterion(outputs, targets)  
                
                scaler.scale(loss).backward()  
                scaler.step(self.optimizer)  
                scaler.update()  
                
                total_loss += loss.item()                 
  
            scheduler.step()  

            # 每隔10次打印一次训练信息和精度   
            if (epoch + 1) % 10 == 0:  
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/len(dataloader):.4f}")  

            
    def evaluate_model(self, dataloader):  
        self.fc.eval()  
        all_targets = []  
        all_predictions = []  

        with torch.no_grad():  
            for images, targets, _ in dataloader:  
                images = images.to(self.device)  
                targets = targets.to(self.device).float()  

                with torch.cuda.amp.autocast():  
                    features = self.get_image_features(images)  
                    outputs = self.fc(features)  
                
                all_targets.append(targets.cpu().numpy())  
                all_predictions.append(outputs.sigmoid().cpu().numpy())  

        all_targets = np.concatenate(all_targets, axis=0)  
        all_predictions = np.concatenate(all_predictions, axis=0)  
        
        threshold = 0.5  
        all_predictions_bin = (all_predictions > threshold).astype(int)  
        f1 = f1_score(all_targets, all_predictions_bin, average='weighted')  
        average_precision = average_precision_score(all_targets, all_predictions, average='weighted')  
        roc_auc = roc_auc_score(all_targets, all_predictions, average='weighted')  
        
        print(f"F1 Score: {f1}")  
        print(f"Average Precision: {average_precision}")  
        print(f"ROC-AUC: {roc_auc}")  
        
        return f1, average_precision, roc_auc  

    def save_model(self, path):  
        torch.save({  
            'model_state_dict': self.model.state_dict(),  
            'fc_state_dict': self.fc.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict()  
        }, path)  

    def load_model(self, path, num_labels):  
        checkpoint = torch.load(path)  
        self.model.load_state_dict(checkpoint['model_state_dict'])  

        self.fc = nn.Sequential(  
            nn.Linear(self.model.visual.output_dim, 512),  
            nn.ReLU(),  
            nn.Linear(512, 256),  
            nn.ReLU(),  
            nn.Linear(256, num_labels)  
        ).to(self.device)  
        self.fc.load_state_dict(checkpoint['fc_state_dict'])  

        # 重新初始化优化器  
        self.optimizer = optim.AdamW(self.fc.parameters())  
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

    def classify_images_from_folder(self, folder_path, output_csv):  
        self.fc.eval()  
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)  
                       if os.path.isfile(os.path.join(folder_path, f))]  
        
        all_predictions = []  
        image_paths = []  
        
        for image_path in image_files:  
            image = Image.open(image_path).convert('RGB')  
            image = self.preprocess_func(image).unsqueeze(0).to(self.device)  # 使用remoteclip的预处理并添加batch dimension  

            with torch.no_grad(), torch.cuda.amp.autocast():  
                features = self.get_image_features(image)  
                outputs = self.fc(features)  
                probs = outputs.sigmoid().cpu().numpy()  

            image_paths.append(image_path)  
            all_predictions.append(probs[0])  
        
        df = pd.DataFrame(all_predictions, columns=[f'label{i}' for i in range(len(all_predictions[0]))])  
        df['image_path'] = image_paths  
        df.to_csv(output_csv, index=False)