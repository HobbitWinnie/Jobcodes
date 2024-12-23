import os  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader, Dataset  
import numpy as np  
import rasterio  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, accuracy_score  
import logging  

# Set up logging  
logging.basicConfig(level=logging.INFO)  

class RemoteSensingDataset(Dataset):  
    def __init__(self, images, labels, transform=None):  
        self.images = images  
        self.labels = labels  
        self.transform = transform  

    def __len__(self):  
        return len(self.images)  

    def __getitem__(self, idx):  
        sample = self.images[idx]  
        label = self.labels[idx]  
        if self.transform:  
            sample = self.transform(sample)  
        return sample, label  

def load_data(image_path, label_path):  
    with rasterio.open(image_path) as src:  
        image = src.read()  # Shape [C, H, W]  
    with rasterio.open(label_path) as src:  
        labels = src.read(1)  # Shape [H, W]  
    return image, labels  

def prepare_dataset(image, labels, patch_size=7):  
    c, h, w = image.shape  
    pad_size = patch_size // 2  
    padded_image = np.pad(image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='edge')  
    X, y = [], []  
    
    for row in range(h):  
        for col in range(w):  
            patch = padded_image[:, row:row + patch_size, col:col + patch_size]  
            label = labels[row, col]  
            X.append(patch)  
            y.append(label)  
    
    return np.array(X), np.array(y)  

class SimpleCNN(nn.Module):  
    def __init__(self, num_classes):  
        super(SimpleCNN, self).__init__()  
        self.conv_layers = nn.Sequential(  
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )  
        self.fc_layers = nn.Sequential(  
            nn.Linear(64 * 1 * 1, 128),  # Adjust based on patch size  
            nn.ReLU(),  
            nn.Linear(128, num_classes)  
        )  

    def forward(self, x):  
        x = self.conv_layers(x)  
        x = x.view(x.size(0), -1)  # Flatten  
        x = self.fc_layers(x)  
        return x  

def train_model(model, train_loader, val_loader, num_epochs=10):  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  

    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        for inputs, labels in train_loader:  
            inputs, labels = inputs.float(), labels.long()  
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')  

        # Validation  
        model.eval()  
        all_preds = []  
        all_labels = []  
        with torch.no_grad():  
            for inputs, labels in val_loader:  
                inputs, labels = inputs.float(), labels.long()  
                outputs = model(inputs)  
                _, predicted = torch.max(outputs, 1)  
                all_preds.extend(predicted.cpu().numpy())  
                all_labels.extend(labels.cpu().numpy())  

        acc = accuracy_score(all_labels, all_preds)  
        logging.info(f'Validation Accuracy: {acc * 100:.2f}%')  
        logging.info(f'Classification Report:\n{classification_report(all_labels, all_preds)}')  

def classify_image(model, image_path, output_path, patch_size=7):  
    model.eval()  
    device = next(model.parameters()).device  
    with rasterio.open(image_path) as src:  
        image = src.read()  # Shape [C, H, W]  

    pad_size = patch_size // 2  
    padded_image = np.pad(image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='edge')  
    h, w = image.shape[1], image.shape[2]  
    result_image = np.zeros((h, w), dtype=np.uint8)  

    with torch.no_grad():  
        for row in range(h):  
            for col in range(w):  
                patch = padded_image[:, row:row + patch_size, col:col + patch_size]  
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  
                output = model(patch_tensor)  
                pred = output.argmax(dim=1).item()  
                result_image[row, col] = pred  

    # Save the classified result  
    with rasterio.open(image_path) as src:  
        profile = src.profile  
        profile.update(dtype=rasterio.uint8, count=1)  

    with rasterio.open(output_path, 'w', **profile) as dst:  
        dst.write(result_image, 1)  
        
    logging.info(f"Classification result saved to {output_path}")  

# Example of usage:  
IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
train_img_path = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
label_img_path = os.path.join(IMAGE_ROOT, 'GF2_train_label.tif')  

# Load data  
image, labels = load_data(train_img_path, label_img_path)  

# Prepare dataset  
X, y = prepare_dataset(image, labels)  

# Split data  
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  

# Create datasets  
train_dataset = RemoteSensingDataset(X_train, y_train)  
val_dataset = RemoteSensingDataset(X_val, y_val)  

# Create data loaders  
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  

# Initialize model  
num_classes = len(np.unique(labels))  
model = SimpleCNN(num_classes=num_classes).to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available  

# Train the model  
train_model(model, train_loader, val_loader, num_epochs=10)  

# Classify an image  
test_img_path = os.path.join(IMAGE_ROOT, 'train_mask.tif')  
output_dir = '/home/Dataset/nw/Segmentation/CpeosTest/result'  
os.makedirs(output_dir, exist_ok=True)  

classify_image(model, test_img_path, os.path.join(output_dir, 'classification_results.tif'))