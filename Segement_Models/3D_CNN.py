import os  
import numpy as np  
import torch  
import random  
import logging  
import rasterio  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split  
from torch.cuda.amp import autocast, GradScaler  
import torchvision.transforms as T  

# Setup logging configuration  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

transform = T.Compose([  
    T.RandomHorizontalFlip(),  
    T.RandomVerticalFlip(),  
    T.RandomRotation(10),  # Rotate images by [-10, 10] degrees  
])   

class RemoteSensingDataset(Dataset):  
    """Custom Dataset for loading remote sensing data."""    
    def __init__(self, images, labels=None):  
        self.images = images  
        self.labels = labels  

    def __len__(self):  
        return len(self.images)  

    def __getitem__(self, idx):                    
        image = self.images[idx]  

        if self.labels is not None:  
            label = self.labels[idx]  
            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)  
        
        return torch.tensor(image, dtype=torch.float32)  
    
class RemoteSensingClassifier(nn.Module):  
    """Convolutional Neural Network for Remote Sensing Image Classification."""  
    
    def __init__(self, num_classes=10):  
        super(RemoteSensingClassifier, self).__init__()  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        self.conv_layers = nn.Sequential(  
            nn.Conv3d(4, 64, kernel_size=3, padding=1),  
            nn.ReLU(),  
            nn.Conv3d(64, 64, kernel_size=3, padding=1),  
            nn.BatchNorm3d(64),  # Batch Normalization  
            nn.ReLU(),  
            nn.MaxPool3d(kernel_size=(2, 2, 1)),  
            nn.Dropout(0.25),  
            nn.Conv3d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm3d(128),  # Batch Normalization  
            nn.ReLU(),  
            nn.Conv3d(128, 128, kernel_size=3, padding=1),  
            nn.BatchNorm3d(128),  # Batch Normalization  
            nn.ReLU(),  
            nn.MaxPool3d(kernel_size=(2, 2, 1)),  
            nn.Dropout(0.25)  
        )  
        self.fc_layers = nn.Sequential(  
            nn.Flatten(),  
            nn.Linear(128, 256),  
            nn.ReLU(),  
            nn.Dropout(0.5),  
            nn.Linear(256, num_classes)  
        )  
    
        # Move the model to the device  
        self.to(self.device)  

    def forward(self, x):  
        x = self.conv_layers(x)  
        x = self.fc_layers(x)  
        return x  

def train_model(model, train_loader, val_loader, model_save_path, epochs=10, lr=0.0001):  
    """Train the model."""      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 Regularization  
    scaler = GradScaler()  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  

    logging.info("Starting model training...")  
    
    for epoch in range(epochs):  
        model.train()  
        total_loss = 0  
        for images, labels in train_loader:  
            images, labels = images.to(device), labels.to(device)  
            
            optimizer.zero_grad()                  
            # Mixed Precision Training  
            with autocast():  
                outputs = model(images)  
                loss = criterion(outputs, labels)  

            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  

            total_loss += loss.item()  

        avg_loss = total_loss / len(train_loader)  
        logging.info(f'Epoch {epoch+1}/{epochs}, Average training loss: {avg_loss:.4f}')  

        scheduler.step()  

        if epoch % 10 == 0 or epoch == epochs - 1:  
            validate_model(model, val_loader, criterion, device)  

    model_save = os.path.join(model_save_path, 'model.pth')  
    logging.info("Model training complete.")  
    torch.save(model.state_dict(), model_save)  
    logging.info("Model saved to model.pth")  

def validate_model(model, val_loader, criterion, device):  
    """Validate the model."""  
    model.eval()  
    val_loss = 0  
    correct = 0  
    with torch.no_grad():  
        for images, labels in val_loader:  
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)  
            val_loss += criterion(outputs, labels).item()  
            pred = outputs.argmax(dim=1, keepdim=True)  
            correct += pred.eq(labels.view_as(pred)).sum().item()  

    val_loss /= len(val_loader.dataset)  
    accuracy = 100. * correct / len(val_loader.dataset)  
    logging.info(f'Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')  

def sample_patches(image, labels=None, patch_size=7, sample_size=10000, nodata_value=None, mode='train'):  
    """Sample patches from the image. In 'train' mode, sample randomly; in 'classify' mode, sample every pixel."""  
    logging.info(f"Sampling patches in {mode} mode...")  
    patches = []  
    patches_labels = []  

    band_count, height, width = image.shape  
    offset = patch_size // 2  

    if mode == 'train':  
        # Generate all possible patch positions  
        possible_positions = [  
            (y, x) for y in range(offset, height - offset)  
            for x in range(offset, width - offset)  
            if labels[y, x] != nodata_value  
        ]  

        # Randomly sample a subset of positions  
        sampled_positions = random.sample(possible_positions, min(sample_size, len(possible_positions)))  

        for y, x in sampled_positions:  
            patch = image[:, y-offset:y+offset+1, x-offset:x+offset+1]  
            patches.append(patch)  
            patches_labels.append(labels[y, x])  

    elif mode == 'classify':  
        # Sample every pixel in the image  
        for y in range(offset, height - offset):  
            for x in range(offset, width - offset):  
                patch = image[:, y-offset:y+offset+1, x-offset:x+offset+1]  
                patches.append(patch)  

    logging.info(f"Patch sampling complete: {len(patches)} patches sampled.")  
    return np.array(patches, dtype=np.float32), np.array(patches_labels, dtype=np.float32) if mode == 'train' else None 

def load_images(train_image_path, label_image_path):  
    """Load and return image and label data."""  
    with rasterio.open(train_image_path) as src:  
        train_image_data = src.read()  

    with rasterio.open(label_image_path) as src:  
        label_image_data = src.read(1)  
        nodata_value = src.nodata  

    return train_image_data, label_image_data, nodata_value  

def load_and_prepare_samples(X_train_sample_path, y_train_sample_path, train_img_path, label_img_path):  
    """Load or generate training samples."""  
    if os.path.exists(X_train_sample_path) and os.path.exists(y_train_sample_path):  
        X_train_sample = np.load(X_train_sample_path)  
        y_train_sample = np.load(y_train_sample_path)  
        logging.info("Sample data loaded successfully.")  

    else:  
        image_data, label_data, nodata_value = load_images(train_img_path, label_img_path)  
        X_train_sample, y_train_sample = sample_patches(image_data, label_data, sample_size=10000, nodata_value=nodata_value,  mode='train')  
        np.save(X_train_sample_path, X_train_sample)  
        np.save(y_train_sample_path, y_train_sample)  

        logging.info("Sample data generated and saved successfully.")  

    return X_train_sample, y_train_sample  

def get_optimal_batch_size_and_workers():  
    """Determine optimal batch size and number of workers for data loading."""  
    num_gpus = torch.cuda.device_count()  
    num_cpus = os.cpu_count()  

    batch_size_per_gpu = 32  
    batch_size = batch_size_per_gpu * max(1, num_gpus)  
    num_workers = min(4 * num_cpus, 42)  

    logging.info(f"Calculated optimal batch_size: {batch_size}, num_workers: {num_workers}")  
    return batch_size, num_workers  

def load_model(model_path, num_classes=10):  
    """Load a trained model from the given file path."""  
    model = RemoteSensingClassifier(num_classes=num_classes)  
    
    # Wrap model if using DataParallel  
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)  

    # Load model state_dict directly  
    model.load_state_dict(torch.load(model_path))  
    model.eval()  # Set the model to evaluation mode  
    logging.info(f"Model loaded from {model_path}")  
    return model

def classify_image(model, image_path, output_path, patch_size=7):  
    """Classify an image and save the result."""  
    model.eval()  # Ensure the model is in evaluation mode  
    logging.info("Starting image classification...")  

    # Check if model is wrapped in DataParallel  
    if isinstance(model, nn.DataParallel):  
        device = model.module.device  
    else:  
        device = model.device

    with rasterio.open(image_path) as src:  
        image_to_classify = src.read()  # Read the image channels  
        logging.info(f'Nodata value for classification image: {src.nodata}')  

    _, height, width = image_to_classify.shape  
    patches, _ = sample_patches(image_to_classify, patch_size=patch_size, mode='classify')  

    # Ensure patches are cropped correctly with appropriate dimensions  
    patches = patches[:, :, :patch_size, :patch_size]  # Assuming 7x7 patches 
    # Convert to torch tensor  
    patches = torch.tensor(patches, dtype=torch.float32).unsqueeze(2).to(device)  # Shape [B, D, C, H, W]  

    predictions = []  
    with torch.no_grad():  
        for patch in patches:  
            patch = patch.unsqueeze(0)  # Add batch dimension [1, D, C, H, W]  
            if patch.size(-2) < 3 or patch.size(-1) < 3:  
                raise ValueError(f"Patch size is too small post processing: {patch.size()}")  

            output = model(patch)  
            pred = output.argmax(dim=1, keepdim=True)  
            predictions.append(pred.item())  

    result_image = np.array(predictions).reshape(height - 2 * (patch_size // 2), width - 2 * (patch_size // 2))  
    logging.info("Image classification complete.")  

    # Save the classified result  
    with rasterio.open(image_path) as src:  
        output_profile = src.profile  
        output_profile.update(dtype=rasterio.uint8, count=1)  

    with rasterio.open(output_path, 'w', **output_profile) as dst:  
        dst.write(result_image.astype(rasterio.uint8), 1)  

    logging.info(f"Classified image saved to {output_path}")  

def main(X_train_sample_path, y_train_sample_path, test_image_path, model_save_path, output_dir):  
    logging.info("Preparing data and model...")  

    X_train_sample, y_train_sample = load_and_prepare_samples(X_train_sample_path, y_train_sample_path, train_img_path, label_img_path)  
    assert X_train_sample.shape[1:] == (4, 7, 7), "Patch dimensions are incorrect."  

    X_train_sample = X_train_sample[..., np.newaxis]  
    X_train, X_val, y_train, y_val = train_test_split(X_train_sample, y_train_sample, test_size=0.8, random_state=42)  

    train_dataset = RemoteSensingDataset(X_train, y_train)  
    val_dataset = RemoteSensingDataset(X_val, y_val)  
    
    batch_size, num_workers = get_optimal_batch_size_and_workers()  
    
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=21, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=21, shuffle=True)  

    model = RemoteSensingClassifier(num_classes=10)  
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)   
    
    train_model(model, train_loader, val_loader, model_save_path, epochs=10)  

    # Load the trained model  
    model = load_model(os.path.join(model_save_path, 'model.pth'))  

    # Classify the test image  
    classify_image(model, test_image_path, os.path.join(output_dir, 'classified_image.tif'))  

if __name__ == "__main__":  
    # Set paths or configure them through function arguments or a config file  
    output_dir = '/home/Dataset/nw/Segmentation/CpeosTest/result'  
    model_save_path = '/home/nw/Codes/Segement_Models/model_save'  
    os.makedirs(model_save_path, exist_ok=True)  
    os.makedirs(output_dir, exist_ok=True)  

    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    train_img_path = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    label_img_path = os.path.join(IMAGE_ROOT, 'GF2_train_label.tif')  
    test_img_path = os.path.join(IMAGE_ROOT, 'train_mask.tif')  
    
    SAMPLE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/samples'  
    X_train_sample_path = os.path.join(SAMPLE_ROOT, 'X_train_sample.npy')  
    y_train_sample_path = os.path.join(SAMPLE_ROOT, 'Y_train_sample.npy')  

    main(X_train_sample_path, y_train_sample_path, test_img_path, model_save_path, output_dir)