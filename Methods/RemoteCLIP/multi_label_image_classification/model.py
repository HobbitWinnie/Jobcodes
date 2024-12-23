# import os  
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"  

import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np  
import pandas as pd  
import open_clip  
from PIL import Image  
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, fbeta_score
from torch.optim.lr_scheduler import StepLR  
from adabelief_pytorch import AdaBelief  
import time  


class MultiLabelClassifierPro:  
    def __init__(self, num_classes, ckpt_path, model_name='ViT-L-14', device=None):  
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  
        self.num_classes = num_classes  

        self.model, _, preprocess_func = open_clip.create_model_and_transforms(model_name, pretrained=False)  
        self.preprocess_func = preprocess_func          
        ckpt = torch.load(ckpt_path, map_location='cpu')  
        self.model.load_state_dict(ckpt)
        self.model.eval()
        self.model = self.model.to(self.device)  

        # 用DataParallel包装模型以支持多GPU  
        if torch.cuda.device_count() > 1:  
            self.model = nn.DataParallel(self.model)  

        # add fully connected layer  
        self.fc = nn.Linear(self.get_visual_output_dim(), num_classes).to(self.device)  
        if torch.cuda.device_count() > 1:  
            self.fc = nn.DataParallel(self.fc)  
       
        self.criterion = nn.BCEWithLogitsLoss()  
        self.optimizer = optim.Adam(self.fc.parameters(), lr=0.001)  
        # self.optimizer = optim.AdamW(self.fc.parameters(), lr=lr)  
        # self.optimizer = optim.RMSprop(self.fc.parameters(), lr=lr, momentum=0.9)  
        # self.optimizer = AdaBelief(self.fc.parameters(), lr=lr, eps=1e-12, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)  

    def get_visual_output_dim(self):  
        if isinstance(self.model, nn.DataParallel):  
            return self.model.module.visual.output_dim  
        else:  
            return self.model.visual.output_dim  
        
    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad():  
            if isinstance(self.model, nn.DataParallel):  
                image_features = self.model.module.encode_image(images)  
            else:  
                image_features = self.model.encode_image(images)  
        
        return image_features / image_features.norm(dim=-1, keepdim=True)  

    def train_model(self, train_loader, val_loader, num_epochs=20):                   
        scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)  
        scaler = torch.cuda.amp.GradScaler()  
        self.fc.train() 

        for epoch in range(num_epochs): 
            total_loss = 0.0  
            start_time = time.time()  # Start time for the epoch  

            for images, targets in train_loader:  
                images, targets = images.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()  
                
                with torch.cuda.amp.autocast():  
                    features = self.get_image_features(images)
                    outputs = self.fc(features)  
                    loss = self.criterion(outputs, targets)  
                
                scaler.scale(loss).backward()  
                scaler.step(self.optimizer)  
                scaler.update()  
                
                total_loss += loss.item()  
            
            scheduler.step()  
            end_time = time.time()  # End time for the epoch  
            epoch_duration = end_time - start_time  # Calculate the durati

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Time: {epoch_duration:.2f} seconds")  
            
            # Evaluate on validation set every 5 epochs  
            if (epoch + 1) % 5 == 0:  
                self.evaluate(val_loader)   
                self.fc.train()  


    # def evaluate_model(self, dataloader):  
    #     self.fc.eval()  
    #     all_targets, all_predictions = [], []  
        
    #     with torch.no_grad():  
    #         for images, targets in dataloader:  
    #             images, targets = images.to(self.device), targets.to(self.device).float()  
                
    #             with torch.cuda.amp.autocast():  
    #                 features = self.get_image_features(images).to(self.device)
    #                 outputs = self.fc(features)  
                
    #             all_targets.append(targets.cpu().numpy())  
    #             all_predictions.append(outputs.sigmoid().cpu().numpy())  
        
    #     all_targets = np.concatenate(all_targets, axis=0)  
    #     all_predictions = np.concatenate(all_predictions, axis=0)  
        
    #     threshold = 0.5  
    #     all_predictions_bin = (all_predictions > threshold).astype(int)  
    #     f1 = f1_score(all_targets, all_predictions_bin, average='weighted', zero_division=1)  
        
    #     print(f"F1 Score: {f1:.4f}")  
    #     return f1  

    def evaluate(self, dataloader):  
        self.model.eval()  
        all_labels = []  
        all_predictions = []

        total_loss = 0.0  
        with torch.no_grad():  
            for images, labels in dataloader:  
                images, labels = images.to(self.device), labels.to(self.device)                  
                features = self.get_image_features(images)
                outputs = self.fc(features)

                loss = self.criterion(outputs, labels)  
                total_loss += loss.item()  
                
                all_labels.extend(labels.cpu().numpy())  
                all_predictions.extend(outputs.sigmoid().cpu().numpy())  
        
        avg_loss = total_loss / len(dataloader)  
        print(f'Validation Loss: {avg_loss}')  

        # Threshold outputs for F1 score calculation  
        thresholded_predictions = [[1 if out >= 0.5 else 0 for out in sample] for sample in all_predictions]         
        
        f1 = f1_score(all_labels, thresholded_predictions, average='macro', zero_division=1)  
        print(f'F1 Score: {f1}')  
       
        f2 = fbeta_score(all_labels, thresholded_predictions, beta=2, average='macro', zero_division=1)  
        print(f'F2 Score: {f2}') 

        return avg_loss  


    # def save_model(self, path):  
    #     torch.save({  
    #         'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),  
    #         'fc_state_dict': self.fc.state_dict(),  
    #         'optimizer_state_dict': self.optimizer.state_dict()  
    #     }, path)  

    # def load_model(self, path, num_labels):  
    #     checkpoint = torch.load(path)  
    #     model_state_dict = checkpoint['model_state_dict']  

    #     if isinstance(self.model, nn.DataParallel):  
    #         self.model.module.load_state_dict(model_state_dict)  
    #     else:  
    #         self.model.load_state_dict(model_state_dict)  
        
    #     self.fc = nn.Linear(self.get_visual_output_dim(), num_labels).to(self.device)  
    #     self.fc.load_state_dict(checkpoint['fc_state_dict'])  

    #     self.optimizer = optim.RMSprop(self.fc.parameters(), lr=1e-4, momentum=0.9)  
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

    #     # self.optimizer = optim.AdamW(self.fc.parameters())  
    #     # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

    # def classify_images_from_folder(self, folder_path, output_csv):  
        self.fc.eval()  
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]  
        
        all_predictions, image_paths = [], []  
        
        for image_path in image_files:  
            try:  
                image = Image.open(image_path).convert('RGB')  
                image = self.preprocess_func(image).unsqueeze(0).to(self.device)  
                
                with torch.no_grad():  
                    features = self.get_image_features(image)  
                    outputs = self.fc(features)  
                
                probs = outputs.sigmoid().cpu().numpy()  
                all_predictions.append(probs[0])  
                image_paths.append(image_path)  
            except Exception as e:  
                print(f"Failed to process image {image_path}: {e}")  
        
        df = pd.DataFrame(all_predictions, columns=[f'label{i}' for i in range(len(all_predictions[0]))])  
        df['image_path'] = image_paths  
        df.to_csv(output_csv, index=False)