import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"   # 指定显卡

import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import models, transforms  
from PIL import Image  
import pandas as pd  
from sklearn.metrics import f1_score  
import torch.optim.lr_scheduler as lr_scheduler  

import time  


class DenseNet201MultiLabelClassifier:  
    def __init__(self, num_classes, device=None):  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  
        self.num_classes = num_classes  
        
        # Load the pre-trained DenseNet201 model  
        self.model = models.densenet201(pretrained=True)  
        
        # Modify the classifier layer for multi-label classification  
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)  
        self.model = self.model.to(self.device)

        if torch.cuda.device_count() > 1:  
            self.model = nn.DataParallel(self.model)
        
        # Define the loss function and optimizer  
        self.criterion = nn.BCEWithLogitsLoss()  
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)  

        # Preprocess function for image transformation  
        self.preprocess_func = transforms.Compose([  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])  

    def train_model(self, train_dataloader, val_dataloader, num_epochs=10):  
        self.model.train()  
        for epoch in range(num_epochs):  
            start_time = time.time()  # Start time for the epoch  
            running_loss = 0.0  
            for inputs, labels in train_dataloader:  
                inputs, labels = inputs.to(self.device), labels.to(self.device)  
                self.optimizer.zero_grad()  
                outputs = self.model(inputs)  
                loss = self.criterion(outputs, labels)  
                loss.backward()  
                self.optimizer.step()  
                running_loss += loss.item()  
        
            end_time = time.time()  # End time for the epoch  
            epoch_duration = end_time - start_time  # Calculate the duration 

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}, Time: {epoch_duration:.2f} seconds')  
            
            # Evaluate on validation set every 5 epochs  
            if (epoch + 1) % 5 == 0:  
                # Evaluate on validation set  
                val_loss = self.evaluate(val_dataloader)   
                # Step the scheduler  
                self.scheduler.step(val_loss)  
                self.model.train()

    def evaluate(self, dataloader, threshold=0.5):  
        self.model.eval()  
        all_labels = []  
        all_predictions = []  
        with torch.no_grad():  
            for inputs, labels in dataloader:  
                inputs, labels = inputs.to(self.device), labels.to(self.device)  
                outputs = self.model(inputs)  
                probabilities = torch.sigmoid(outputs).cpu().numpy()  
                predictions = (probabilities > threshold).astype(int)  
                
                all_labels.extend(labels.cpu().numpy())  
                all_predictions.extend(predictions)  

        # Calculate F1 score  
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)  
        print(f'F1 Score: {f1}')  
        return f1  

    def classify_image(self, image_path):  
        try:  
            image = Image.open(image_path).convert("RGB")  
        except Exception as e:  
            print(f"Error opening image {image_path}: {e}")  
            return None  

        image_tensor = self.preprocess_func(image).unsqueeze(0).to(self.device)  
        self.model.eval()  
        with torch.no_grad():  
            output = self.model(image_tensor)  
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()  
        return probabilities  

    def classify_images_in_folder(self, folder_path, output_csv):  
        results = []  
        for img_name in os.listdir(folder_path):  
            img_path = os.path.join(folder_path, img_name)  
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                probabilities = self.classify_image(img_path)  
                if probabilities is not None:  
                    results.append({"filename": img_name, "probabilities": probabilities})  

        df = pd.DataFrame(results)  
        df.to_csv(output_csv, index=False)  
        print(f"Results saved to `{output_csv}`")  