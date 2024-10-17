import os  
import torch  
from torch.utils.data import DataLoader  
import rasterio  
import numpy as np  
import torch.nn as nn  
import logging  
from torch.optim.lr_scheduler import StepLR  
from torch.cuda.amp import GradScaler, autocast  
import torchvision.transforms as transforms  

from dataset import RemoteSensingDataset, reconstruct_image_from_patches, split_image_into_patches  
from model import UNet  

# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

def load_data(image_path, label_path=None):  
    logging.info(f"Loading image from {image_path}")  
    with rasterio.open(image_path) as src:  
        image = src.read()  
        image_nodata = int(src.nodata)  
        image_meta = src.meta  

    image_mask = (image[0] != image_nodata)  

    if label_path:  
        logging.info(f"Loading labels from {label_path}")  
        with rasterio.open(label_path) as src:  
            labels = src.read(1)  
            labels_nodata = int(src.nodata)  

        label_mask = (labels != labels_nodata)  
        labels = np.where(label_mask, labels, 0)  
    else:  
        labels = None  
        labels_nodata = 0  

    image = np.where(image_mask, image, 0)  
    return image, labels, image_mask, labels_nodata, image_meta  

def train(model, train_loader, device, epochs, learning_rate, save_path):  
    model.to(device)  
    criterion = nn.CrossEntropyLoss(reduction='none')  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  
    scaler = GradScaler()  

    for epoch in range(epochs):  
        model.train()  
        total_loss = 0  

        for batch in train_loader:  
            img_patch, label_patch, mask_patch = batch  
            img_patch, label_patch, mask_patch = img_patch.to(device), label_patch.to(device), mask_patch.to(device)  
            optimizer.zero_grad()  

            with autocast():  
                outputs = model(img_patch)  
                loss = criterion(outputs, label_patch.long())  
                masked_loss = (loss * mask_patch.unsqueeze(1)).sum() / mask_patch.sum()  

            scaler.scale(masked_loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  

            total_loss += masked_loss.item()  

        average_loss = total_loss / len(train_loader)  
        logging.info(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.4f}")  

        scheduler.step()  

    torch.save(model.state_dict(), save_path)  
    logging.info(f"Model saved to {save_path}")  

def predict(model, save_path, test_image_paths, output_paths, patch_size, overlap, device):  
    model.load_state_dict(torch.load(save_path, map_location=device))  
    model.eval()  

    for test_image_path, output_path in zip(test_image_paths, output_paths):  
        logging.info(f"Predicting for {test_image_path}")  
        test_image, _, test_mask, _, image_profile = load_data(test_image_path)  
        patches = split_image_into_patches(test_image, patch_size, overlap)  
        predictions = []  

        with torch.no_grad():  
            for patch in patches:  
                patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  
                output = model(patch)  
                pred = output.argmax(dim=1).squeeze().cpu().numpy()  
                predictions.append(pred)  

        reconstructed_prediction = reconstruct_image_from_patches(predictions, test_image.shape, patch_size, overlap)  
        reconstructed_prediction = np.where(test_mask, reconstructed_prediction, 0)  

        # Save prediction  
        image_profile.update(dtype=rasterio.float32, count=1)  
        with rasterio.open(output_path, 'w', **image_profile) as dst:  
            dst.write(reconstructed_prediction, 1)  
        logging.info(f"Prediction saved to {output_path}")  

def main():  
    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    IMAGE_PATH = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    LABEL_PATH = os.path.join(IMAGE_ROOT, 'train_label.tif')  

    save_path = '/home/nw/Codes/Segement_Models/model_save/model_gptUNet.pth'  
    test_img_paths = [  
        os.path.join(IMAGE_ROOT, 'train_mask.tif'),  
        os.path.join(IMAGE_ROOT, 'GF2_test_image.tif')  
    ]  
    output_paths = [  
        '/home/Dataset/nw/Segmentation/CpeosTest/result/train_mask_gptUnet_results.tif',  
        '/home/Dataset/nw/Segmentation/CpeosTest/result/GF2_test_image_gptUnet_results.tif'  
    ]  

    PATCH_SIZE = 256  
    PATCH_NUMBER = 10000  
    OVERLAP = 32  
    BATCH_SIZE = 192  # Adjust if needed based on GPU memory  
    EPOCHS = 1000  
    LEARNING_RATE = 0.001  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Initialize model  
    logging.info("Initializing UNet model")  
    model = UNet(in_channels=4, out_channels=10, dropout_rate=0.1)  
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)  
    model.to(device)  

    # Load training data  
    image, labels, _, labels_nodata, _ = load_data(IMAGE_PATH, LABEL_PATH)  

    train_dataset = RemoteSensingDataset(image, labels, labels_nodata, patch_size=PATCH_SIZE, num_patches=PATCH_NUMBER)  
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)  

    # Train the model  
    train(model, train_loader, device, EPOCHS, LEARNING_RATE, save_path)  

    # Predict and save results  
    predict(model, save_path, test_img_paths, output_paths, PATCH_SIZE, OVERLAP, device)  

if __name__ == "__main__":  
    main()