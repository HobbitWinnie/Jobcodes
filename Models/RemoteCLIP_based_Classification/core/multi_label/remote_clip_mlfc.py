import os  
import torch  
import logging  
import time  
import open_clip  
import numpy as np  
import pandas as pd  
from torch import nn, optim  
from datetime import datetime  
from PIL import Image  
from sklearn.metrics import f1_score, fbeta_score  
from torch.optim.lr_scheduler import OneCycleLR  


# Set up logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  


class RemoteCLIPClassifierFC:  
    def __init__(self, ckpt_path, num_labels, model_name='ViT-L-14', device=None):  
        self.model_name = model_name  
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  
        
        # Load CLIP model and preprocessing function  
        self.model, _, preprocess_func = open_clip.create_model_and_transforms(self.model_name)  
        ckpt = torch.load(ckpt_path, map_location='cpu')  
        self.model.load_state_dict(ckpt)  
        self.model = self.model.to(self.device).eval()  
        
        # Freeze all layers except the last few  
        for param in self.model.parameters():  
            param.requires_grad = False  
        
        # Initialize the final classifier layer with BatchNorm and Dropout  
        self.fc = nn.Sequential(  
            nn.Linear(self.model.visual.output_dim, 512),  
            nn.BatchNorm1d(512),  
            nn.ReLU(),  
            nn.Dropout(0.5),  
            nn.Linear(512, num_labels)  
        ).to(self.device)  
        
        self.criterion = nn.BCEWithLogitsLoss()  
        self.optimizer = optim.AdamW(self.fc.parameters(), lr=0.001)  
        self.scheduler = None  # Will be initialized in train_model  

        # Set the preprocess function  
        self.preprocess_func = preprocess_func  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad(), torch.cuda.amp.autocast():  
            image_features = self.model.encode_image(images)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features  

    def train_model(self, train_dataloader, val_dataloader, num_epochs=50):   
        current_time = datetime.now().strftime('%H:%M:%S')   
        logger.info("Start training RemoteCLIP_FC. Time: {}".format(current_time))  
        scaler = torch.cuda.amp.GradScaler()  
        
        # Initialize OneCycleLR scheduler  
        self.scheduler = OneCycleLR(self.optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=num_epochs)  
        
        self.fc.train()  
        for epoch in range(num_epochs):  
            epoch_start_time = time.time()  
            total_loss = 0.0  
            for images, targets in train_dataloader:  
                images, targets = images.to(self.device), targets.to(self.device).float()  

                self.optimizer.zero_grad()  
                
                with torch.cuda.amp.autocast():  
                    features = self.get_image_features(images)  
                    outputs = self.fc(features)  
                    loss = self.criterion(outputs, targets)  
                
                scaler.scale(loss).backward()  
                scaler.step(self.optimizer)  
                scaler.update()  
                
                total_loss += loss.item()  
                self.scheduler.step()  # Update learning rate  

            avg_loss = total_loss / len(train_dataloader)  
            epoch_duration = time.time() - epoch_start_time  
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s')  

            # Validate every 5 epochs if validation data is provided  
            if val_dataloader and (epoch + 1) % 5 == 0:  
                self.evaluate_model(val_dataloader)  
                self.fc.train()  
       
        current_time = datetime.now().strftime('%H:%M:%S')   
        logger.info("RemoteCLIP_FC training compeleted. Time: {}".format(current_time)) 

    def evaluate_model(self, dataloader):  
        self.fc.eval()  
        all_targets, all_predictions = [], []  
          
        with torch.no_grad():  
            for images, targets in dataloader:  
                images = images.to(self.device)  
                targets = targets.cpu().numpy()  
                
                with torch.cuda.amp.autocast():  
                    features = self.get_image_features(images)  
                    outputs = self.fc(features)  
                
                all_targets.append(targets)  
                all_predictions.append(outputs.sigmoid().cpu().numpy())  
        
        all_targets = np.concatenate(all_targets, axis=0)  
        all_predictions = np.concatenate(all_predictions, axis=0)  
        
        threshold = 0.5  
        all_predictions_bin = (all_predictions > threshold).astype(int)  

        f1 = f1_score(all_targets, all_predictions_bin, average='macro', zero_division=1)  
        f2 = fbeta_score(all_targets, all_predictions_bin, beta=2, average='macro', zero_division=1)  

        logger.info(f'Validation - F1 Score: {f1:.4f}, F2 Score: {f2:.4f}')  
        
        return f1, f2  

    def save_model(self, path):  
        torch.save({  
            'model_state_dict': self.model.state_dict(),  
            'fc_state_dict': self.fc.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict()  
        }, path)  
        logger.info(f'模型已保存到 {path}')  

    def load_model(self, path, num_labels):  
        checkpoint = torch.load(path, map_location=self.device)  
        self.model.load_state_dict(checkpoint['model_state_dict'])  
        
        self.fc = nn.Sequential(  
            nn.Linear(self.model.visual.output_dim, 512),  
            nn.BatchNorm1d(512),  
            nn.ReLU(),  
            nn.Dropout(0.5),  
            nn.Linear(512, num_labels)  
        ).to(self.device)  
        self.fc.load_state_dict(checkpoint['fc_state_dict'])  
        
        self.optimizer = optim.AdamW(self.fc.parameters())  
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        logger.info(f'模型已加载自 {path}')  

    def classify_images_from_folder(self, folder_path, output_csv):  
        self.fc.eval()  
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')  
        
        all_predictions = []  
        image_paths = []  
        
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if not img_path.lower().endswith(valid_extensions):  
                continue  
            
            try:  
                image = Image.open(img_path).convert('RGB')  
                image = self.preprocess_func(image).unsqueeze(0).to(self.device)  # 使用预处理函数  
                
                with torch.no_grad(), torch.cuda.amp.autocast():  
                    features = self.get_image_features(image)  
                    outputs = self.fc(features)  
                    probs = outputs.sigmoid().cpu().numpy()  
                
                image_paths.append(img_path)  
                all_predictions.append(probs[0])  
            except Exception as e:  
                logger.error(f"处理图像 {img_name} 时出错: {e}")  

        df = pd.DataFrame(all_predictions, columns=[f'label{i}' for i in range(len(all_predictions[0]))])  
        df['image_path'] = image_paths  
        df.to_csv(output_csv, index=False)  
        logger.info(f'预测结果已保存到 {output_csv}')  