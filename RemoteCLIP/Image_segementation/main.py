import os  
import torch  
import logging  
import rasterio  

import numpy as np  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  

from model import load_clip_model, RemoteCLIPSegmentation  
from data_utils import load_data, sample_dataset, load_dataset, save_dataset, SegmentationDataset
from train import train_model


# Set up logging  
logging.basicConfig(level=logging.INFO)  

def classify_large_image(model, image_path, output_path, patch_size=256, no_data_value=-1):  
    """Classify a large image using a trained model."""  
    model.eval()  
    device = next(model.parameters()).device  

    with rasterio.open(image_path) as src:  
        image = src.read()  # Shape [C, H, W]  
        profile = src.profile  

    h, w = image.shape[1], image.shape[2]  
    result_image = np.full((h, w), no_data_value, dtype=np.float32)  

    pad_size = patch_size // 2  
    padded_image = np.pad(image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='reflect')  

    with torch.no_grad():  
        for row in range(0, h, patch_size):  
            for col in range(0, w, patch_size):  
                patch = padded_image[:, row:row + patch_size + pad_size, col:col + patch_size + pad_size]  
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  
                output = model(patch_tensor)  
                pred = output.argmax(dim=1).squeeze().cpu().numpy()  
                result_image[row:row + patch_size, col:col + patch_size] = pred[:patch_size, :patch_size]  

    profile.update(dtype=rasterio.float32, count=1)  

    with rasterio.open(output_path, 'w', **profile) as dst:  
        dst.write(result_image, 1)  

    print(f"Classification result saved to {output_path}")  


def prepare_data(X_path, y_path, image_path, label_path, patch_size):  
        # Load data  
        image, labels, nodata_value = load_data(image_path, label_path)  

        # Prepare dataset  
        X, y = load_dataset(X_path, y_path)  
        if X is None or y is None:  
            X, y = sample_dataset(image, labels, 50000, 11)  
            save_dataset(X, y, X_path, y_path)  

        # Split data  
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)  

        # Create datasets and loaders  
        train_dataset = SegmentationDataset(X_train, y_train, patch_size)  
        val_dataset = SegmentationDataset(X_val, y_val, patch_size)  
        train_loader = DataLoader(train_dataset, batch_size=192, num_workers=42, shuffle=True)  
        val_loader = DataLoader(val_dataset, batch_size=192, num_workers=42, shuffle=False)  

        return train_loader, val_loader  


if __name__ == "__main__":  
    # Set paths  
    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    train_img_path = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    label_img_path = os.path.join(IMAGE_ROOT, 'train_label.tif')  
    
    SAMPLE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/samples'  
    X_path = os.path.join(SAMPLE_ROOT, 'X_sample_11_50000.npy')  
    y_path = os.path.join(SAMPLE_ROOT, 'Y_sample_11_50000.npy')  
    save_path = '/home/nw/Codes/Segement_Models/model_save/model.pth'

    test_img_path_1 = os.path.join(IMAGE_ROOT, 'train_mask.tif')      
    output_path_1 = '/home/Dataset/nw/Segmentation/CpeosTest/result/train_mask_Rere50_results.tif'  

    test_img_path_2 = os.path.join(IMAGE_ROOT, 'GF2_test_image.tif')  
    output_path_2 = '/home/Dataset/nw/Segmentation/CpeosTest/result/GF2_test_image_Rere50_results.tif'  

    # Load trained CLIP model  
    clip_ckpt_path = '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-RN50.pt'  
    model_name = "RN50"  
    num_classes = 10  # Adjust based on your dataset  
    seg_model = RemoteCLIPSegmentation(clip_ckpt_path, num_classes, model_name)

    # Load data  
    train_loader, val_loader = prepare_data(X_path, y_path, train_img_path, label_img_path, 256)
    train_model(seg_model, train_loader, val_loader, num_classes, 1000)  

    classify_large_image(seg_model, test_img_path_1, output_path_1, patch_size=256, no_data_value=15)  
    classify_large_image(seg_model, test_img_path_2, output_path_2, patch_size=256, no_data_value=15)  

