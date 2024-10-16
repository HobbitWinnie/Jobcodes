import os  
import torch  
from torch.utils.data import DataLoader  
import rasterio  
import numpy as np  
import torch.nn as nn  
import logging  

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
        logging.info(f"Image nodata value: {image_nodata}")  
        logging.info(f"Image profile: {image_meta}")  

    image_mask = (image[0] != image_nodata)   

    if label_path:  
        logging.info(f"Loading labels from {label_path}")  
        with rasterio.open(label_path) as src:  
            labels = src.read(1)  
            labels_nodata = int(src.nodata)  
            logging.info(f"Labels nodata value: {labels_nodata}")  

        label_mask = (labels != labels_nodata)  
        labels = np.where(label_mask, labels, 0)  
    else:  
        labels = None  
        labels_nodata = 0  

    image = np.where(image_mask, image, 0)  
    return image, labels, image_mask, labels_nodata, image_meta

def prepare_batch(batch, device):  
    img_patch, label_patch, mask_patch = batch  
    img_patch = img_patch.to(device)  
    label_patch = label_patch.to(device)  
    mask_patch = mask_patch.to(device)  
    return img_patch, label_patch, mask_patch  

def compute_masked_loss(outputs, targets, mask, criterion):  
    loss = criterion(outputs, targets.long())  
    masked_loss = loss * mask.unsqueeze(1)  # Ensure mask is broadcastable  
    final_loss = masked_loss.sum() / mask.sum()  # Normalize by effective number of valid elements  
    return final_loss  

def train(model, train_loader, device, epochs, learning_rate, save_path):  
    model.to(device)  
    criterion = torch.nn.CrossEntropyLoss(reduction='none')  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    
    for epoch in range(epochs):  
        model.train()  
        total_loss = 0  

        for batch in train_loader:  
            img_patch, label_patch, mask_patch = prepare_batch(batch, device)  
            optimizer.zero_grad()  
            
            outputs = model(img_patch)  
            loss = compute_masked_loss(outputs, label_patch, mask_patch, criterion)  
            
            loss.backward()  
            optimizer.step()  

            total_loss += loss.item()  

        average_loss = total_loss / len(train_loader)  
        logging.info(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.4f}")  

    torch.save(model.state_dict(), save_path)  
    logging.info(f"Model saved to {save_path}")  

def predict(model, image, mask, patch_size, overlap, device):  
    model.eval()  
    logging.info("Starting prediction")  
    patches = split_image_into_patches(image, patch_size, overlap)  
    predictions = []  
    
    with torch.no_grad():  
        for i, patch in enumerate(patches):  
            patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  
            output = model(patch)  
            pred = output.argmax(dim=1).squeeze().cpu().numpy()  
            predictions.append(pred)  
            logging.debug(f"Processed patch {i+1}/{len(patches)}")  
            
    reconstructed_prediction = reconstruct_image_from_patches(predictions, image.shape, patch_size, overlap)  
    reconstructed_prediction = np.where(mask, reconstructed_prediction, 0)  
    logging.info("Prediction complete")  
    return reconstructed_prediction  

def save_prediction(prediction, meta, output_path):  
    # Update metadata for saving  
    meta.update(dtype=rasterio.float32, count=1)  

    # Save the segmented image  
    with rasterio.open(output_path, 'w', **meta) as dst:  
        dst.write(prediction, 1) 

    logging.info(f"Prediction saved to {output_path}")  


def main():  
    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    IMAGE_PATH = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    LABEL_PATH = os.path.join(IMAGE_ROOT, 'train_label.tif')  

    save_path = '/home/nw/Codes/Segement_Models/model_save/model_gptUNet.pth'  
    test_img_path = os.path.join(IMAGE_ROOT, 'train_mask.tif')  
    output_path = '/home/Dataset/nw/Segmentation/CpeosTest/result/train_mask_gptUnet_results.tif'  

    PATCH_SIZE = 256  
    PATCH_NUMBER = 1000
    OVERLAP = 128  
    BATCH_SIZE = 192 
    EPOCHS = 5
    LEARNING_RATE = 0.001  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    # Create and train the model  
    logging.info("Initializing UNet model")  
    model = UNet(in_channels=4, out_channels=10) 
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)  
    model.to(device)   

    # Load training data  
    image, labels, _ , labels_nodata,_ = load_data(IMAGE_PATH, LABEL_PATH)  

    # Create datasets and loaders  
    train_dataset = RemoteSensingDataset(image, labels, labels_nodata, patch_size=PATCH_SIZE, num_patches=PATCH_NUMBER)  
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=42)  

    train(model, train_loader, device, EPOCHS, LEARNING_RATE, save_path)  

    # Load test image and perform prediction  
    test_image, _, test_mask, _ , image_profile = load_data(test_img_path)  
    
    predicted_image = predict(model, test_image, test_mask, PATCH_SIZE, OVERLAP, device)  

    # Save the predicted image  
    save_prediction(predicted_image, image_profile, output_path)  

if __name__ == "__main__":  
    main()