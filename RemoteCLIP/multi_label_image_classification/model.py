import os  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np  
import pandas as pd  
import open_clip  
from PIL import Image  
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score  
from torch.optim.lr_scheduler import StepLR  
import torchvision.transforms as transforms  

class MultiLabelClassifierPro:  
    def __init__(self, ckpt_path, model_name='ViT-L-14', device=None):  
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  
        self.model, _, preprocess_func = open_clip.create_model_and_transforms(model_name, pretrained=False)  
        self.preprocess_func = preprocess_func  
        self.tokenizer = open_clip.get_tokenizer(model_name)  
        
        ckpt = torch.load(ckpt_path, map_location='cpu')  
        self.model.load_state_dict(ckpt)  
        self.model.to(self.device).eval()  
        
        # Freeze pre-trained model weights  
        for param in self.model.parameters():  
            param.requires_grad = False  
        
        self.fc = None  
        self.optimizer = None  

    def get_image_features(self, images):  
        images = images.to(self.device)  
        with torch.no_grad():  
            image_features = self.model.encode_image(images)  
        return image_features / image_features.norm(dim=-1, keepdim=True)  

    def train_model(self, train_loader, val_loader, num_labels, num_epochs=2, lr=1e-4, criterion=None):  
        self.fc = nn.Linear(self.model.visual.output_dim, num_labels).to(self.device)  
        criterion = criterion or nn.BCEWithLogitsLoss()  
        
        self.optimizer = optim.AdamW(self.fc.parameters(), lr=lr)  
        scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)  
        scaler = torch.cuda.amp.GradScaler()  
        
        best_val_f1 = 0.0  
        best_model_path = "/home/nw/Codes/RemoteCLIP/multi_label_image_classification/best_model.pth"  
        
        for epoch in range(num_epochs):  
            self.fc.train()  
            total_loss = 0.0  
            for images, targets in train_loader:  
                images, targets = images.to(self.device), targets.to(self.device).float()  
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
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")  
            
            # Evaluate on validation set  
            val_f1, _, _ = self.evaluate_model(val_loader)  
            
            # Save best model  
            if val_f1 > best_val_f1:  
                best_val_f1 = val_f1  
                self.save_model(best_model_path)  
                print(f"New best model saved with F1: {val_f1:.4f}")  

    def evaluate_model(self, dataloader):  
        self.fc.eval()  
        all_targets, all_predictions = [], []  
        
        with torch.no_grad():  
            for images, targets in dataloader:  
                images, targets = images.to(self.device), targets.to(self.device).float()  
                
                with torch.cuda.amp.autocast():  
                    features = self.get_image_features(images)  
                    outputs = self.fc(features)  
                
                all_targets.append(targets.cpu().numpy())  
                all_predictions.append(outputs.sigmoid().cpu().numpy())  
        
        all_targets = np.concatenate(all_targets, axis=0)  
        all_predictions = np.concatenate(all_predictions, axis=0)  
        
        threshold = 0.5  
        all_predictions_bin = (all_predictions > threshold).astype(int)  
        try:  
            f1 = f1_score(all_targets, all_predictions_bin, average='weighted', zero_division=1)  
            average_precision = average_precision_score(all_targets, all_predictions, average='weighted')  
            
            # Check if more than one class is present before computing ROC AUC  
            if np.unique(all_targets).size > 1:  
                roc_auc = roc_auc_score(all_targets, all_predictions, average='weighted')  
            else:  
                print("Warning: Only one class present in y_true. ROC AUC score is not defined in that case.")  
                roc_auc = float('nan')              
            
            print(f"F1 Score: {f1:.4f}")  
            print(f"Average Precision: {average_precision:.4f}")  
            print(f"ROC-AUC: {roc_auc:.4f}")  
            
            return f1, average_precision, roc_auc  
        except ValueError as e:  
            print(f"Error computing metrics: {e}")  
            return 0.0, 0.0, 0.0  

    def save_model(self, path):  
        torch.save({  
            'model_state_dict': self.model.state_dict(),  
            'fc_state_dict': self.fc.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict()  
        }, path)  

    def load_model(self, path, num_labels):  
        checkpoint = torch.load(path)  
        self.model.load_state_dict(checkpoint['model_state_dict'])  
        
        self.fc = nn.Linear(self.model.visual.output_dim, num_labels).to(self.device)  
        self.fc.load_state_dict(checkpoint['fc_state_dict'])  

        self.optimizer = optim.AdamW(self.fc.parameters())  
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

    def classify_images_from_folder(self, folder_path, output_csv):  
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