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


# 设置日志配置  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  


class RemoteSensingDataset(Dataset):  
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
    def __init__(self, num_classes=10):  
        super(RemoteSensingClassifier, self).__init__()  
        self.conv_layers = nn.Sequential(  
            nn.Conv3d(4, 32, kernel_size=3, padding=1),  
            nn.ReLU(),  
            nn.Conv3d(32, 32, kernel_size=3, padding=1),  
            nn.ReLU(),  
            nn.MaxPool3d(kernel_size=(2, 2, 1)),  
            nn.Dropout(0.25),  
            nn.Conv3d(32, 64, kernel_size=3, padding=1),  
            nn.ReLU(),  
            nn.Conv3d(64, 64, kernel_size=3, padding=1),  
            nn.ReLU(),  
            nn.MaxPool3d(kernel_size=(2, 2, 1)),  
            nn.Dropout(0.25)  
        )  
        self.fc_layers = nn.Sequential(  
            nn.Flatten(),  
            nn.Linear(64, 128),  # Adjust the dimensions based on your input size  
            nn.ReLU(),  
            nn.Dropout(0.5),  
            nn.Linear(128, num_classes)  
        )  

    def forward(self, x):  
        x = self.conv_layers(x)  
        x = self.fc_layers(x)  
        return x  


def extract_patches(image, patch_size=7, stride=1):  
    """Extract all patches from an image."""  
    logging.info("Extracting patches from images...")  
    patches = []  

    band_count, height, width = image.shape  
    offset = patch_size // 2  

    for y in range(offset, height - offset, stride):  
        for x in range(offset, width - offset, stride):  
            patch = image[:, y-offset:y+offset+1, x-offset:x+offset+1]  
            patches.append(patch)  

    logging.info("Patch extraction complete: {} patches extracted.".format(len(patches)))  
    return np.array(patches)  


def sample_patches(image, labels, patch_size=7, sample_size=10000, nodata_value=None):  
    """Sample patches from the image, skipping nodata values."""  
    logging.info("Sampling patches from images...")  
    patches = []  
    patches_labels = []  

    band_count, height, width = image.shape  
    offset = patch_size // 2  

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

    logging.info("Patch sampling complete: {} patches sampled.".format(len(patches)))  
    return np.array(patches), np.array(patches_labels)  


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):  
    """Train the model."""  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model = nn.DataParallel(model)  
    model.to(device)  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    logging.info("Starting model training...")  
    for batch_idx, (images, labels) in enumerate(train_loader):  
        print(f"Batch image shape: {images.shape}")  
        break  # 只打印一个batch形状  

    for epoch in range(epochs):  
        model.train()  
        total_loss = 0  
        for batch_idx, (images, labels) in enumerate(train_loader):  
            images, labels = images.to(device), labels.to(device)  
            optimizer.zero_grad()  
            outputs = model(images)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            total_loss += loss.item()  

            # Log every batch  
            if batch_idx % 10 == 0:  
                logging.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')  

        avg_loss = total_loss / len(train_loader)  
        logging.info(f'Epoch {epoch+1}, Average training loss: {avg_loss:.4f}')  

        # Validation  
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
        logging.info(f'Epoch {epoch+1}, Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')  

    logging.info("Model training complete.")  
    torch.save(model.state_dict(), 'model.pth')  
    logging.info("Model saved to model.pth")  


def classify_image(model, image_path, output_path, patch_size=7):  
    """Classify an image and save the result."""  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model.to(device)  
    model.eval()  

    logging.info("Starting image classification...")  

    with rasterio.open(image_path) as src:  
        image_to_classify = src.read()  
        logging.info(f'Nodata value for classification image: {src.nodata}')  # Logging nodata  

    _, height, width = image_to_classify.shape  
    patches = extract_patches(image_to_classify, patch_size=patch_size)  
    patches = patches[..., np.newaxis]  
    patches = torch.tensor(patches, dtype=torch.float32).unsqueeze(1) / 255.0  

    predictions = []  
    with torch.no_grad():  
        for patch in patches:  
            patch = patch.to(device)  
            output = model(patch.unsqueeze(0))  
            pred = output.argmax(dim=1, keepdim=True)  
            predictions.append(pred.item())  

    result_image = np.array(predictions).reshape(height, width)  
    logging.info("Image classification complete.")  

    with rasterio.open(image_path) as src:  
        output_profile = src.profile  
        output_profile.update(dtype=rasterio.uint8, count=1)  

    with rasterio.open(output_path, 'w', **output_profile) as dst:  
        dst.write(result_image.astype(rasterio.uint8), 1)  

    logging.info(f"Classified image saved to {output_path}")  


def get_optimal_batch_size_and_workers():  
    """Determine optimal batch size and number of workers for data loading."""  
    num_gpus = torch.cuda.device_count()  
    num_cpus = os.cpu_count()  

    batch_size_per_gpu = 32  
    batch_size = batch_size_per_gpu * max(1, num_gpus)  

    num_workers = min(4 * num_cpus, 42)  

    logging.info(f"Calculated optimal batch_size: {batch_size}, num_workers: {num_workers}")  
    return batch_size, num_workers  


def load_images(train_image_path, label_image_path):  
    """Load and return image and label data."""  
    with rasterio.open(train_image_path) as src:  
        train_image_data = src.read()  

    with rasterio.open(label_image_path) as src:  
        label_image_data = src.read(1)  
        nodata_value = src.nodata  

    return train_image_data, label_image_data, nodata_value  


def main(train_image_path, label_image_path, test_image_path, output_dir):  
    logging.info("Preparing data and model...")  

    image_data, label_data, nodata_value = load_images(train_image_path, label_image_path)  

    # Sample patches for training  
    sample_size = 10000  # Set based on your requirements  
    X_train_sample, y_train_sample = sample_patches(image_data, label_data, sample_size=sample_size, nodata_value=nodata_value)  

    X_train_sample = X_train_sample[..., np.newaxis]  
    X_train, X_val, y_train, y_val = train_test_split(X_train_sample, y_train_sample, test_size=0.2, random_state=42)  

    batch_size, num_workers = get_optimal_batch_size_and_workers()  

    train_dataset = RemoteSensingDataset(X_train, y_train)  
    val_dataset = RemoteSensingDataset(X_val, y_val)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)  
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)  

    model = RemoteSensingClassifier(num_classes=10)  
    train_model(model, train_loader, val_loader, epochs=1000)  
    
    classified_result_path = os.path.join(output_dir, 'classified_result.tif')  
    classify_image(model, test_image_path, classified_result_path)  


if __name__ == "__main__":  
    # Set paths or configure them through function arguments or a config file  
    train_img_path = '/home/Dataset/nw/Segmentation/CpeosTest/train/GF2_train_image/GF2_train_image.tif'  
    label_img_path = '/home/Dataset/nw/Segmentation/CpeosTest/train/GF2_train_label/GF2_train_label.tif'  
    test_img_path = '/home/Dataset/nw/Segmentation/CpeosTest/test/GF_test_image/GF2_test_image.tif'  
    results_dir = '/home/Dataset/nw/Segmentation/CpeosTest/result'  
    
    main(train_img_path, label_img_path, test_img_path, results_dir)