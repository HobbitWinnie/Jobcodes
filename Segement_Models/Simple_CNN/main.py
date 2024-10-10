import os  
import torch  
import rasterio  
import logging  
import numpy as np  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  

from data_utils import load_data, sample_dataset, load_dataset, save_dataset, RemoteSensingDataset
from model_CNN import SimpleCNN  
from model_ResNet18 import ResNet18
from model_ResNet50 import ResNet50
from train import train_model  


# Set up logging  
logging.basicConfig(level=logging.INFO)  

def classify_image(model_path, image_path, output_path, no_data_value, patch_size=7):  
    """  
    Classify an image using a trained model, avoiding nodata values.  

    Parameters:  
    - model: the trained PyTorch model  
    - image_path: str, path to the input image  
    - output_path: str, path to save the classified result  
    - patch_size: int, size of the patch to extract (default: 7)  
    - no_data_value: int, the value in labels that represents no data (default: -1)  
    """  
    # Load the trained model  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = ResNet50(num_classes=10).to(device)  
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device)  
    model.eval()  

    with rasterio.open(image_path) as src:  
        image = src.read()  # Shape [C, H, W]  
        profile = src.profile  

    pad_size = patch_size // 2  
    padded_image = np.pad(image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='edge')  
    h, w = image.shape[1], image.shape[2]  
    result_image = np.full((h, w), no_data_value, dtype=np.float32)  # Initialize with nodata  

    with torch.no_grad():  
        for row in range(h):  
            for col in range(w):  
                if image[0, row, col] == no_data_value:  # Assuming the first channel indicates nodata  
                    continue  # Skip nodata pixels  

                patch = padded_image[:, row:row + patch_size, col:col + patch_size]  
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  
                output = model(patch_tensor)  
                pred = output.argmax(dim=1).item()  
                result_image[row, col] = pred  

    # Update profile for output  
    profile.update(dtype=rasterio.float32, count=1)  

    with rasterio.open(output_path, 'w', **profile) as dst:  
        dst.write(result_image, 1)  
        
    print(f"Classification result saved to {output_path}")  


if __name__ == "__main__":  
    
    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    train_img_path = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    label_img_path = os.path.join(IMAGE_ROOT, 'train_label.tif')  

    SAMPLE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/samples'  
    X_path = os.path.join(SAMPLE_ROOT, 'X_sample_11_50000.npy')  
    y_path = os.path.join(SAMPLE_ROOT, 'Y_sample_11_50000.npy')  
    model_path = '/home/nw/Codes/Segement_Models/model_save/model.pth'

    test_img_path_1 = os.path.join(IMAGE_ROOT, 'train_mask.tif')  
    output_path_1 = '/home/Dataset/nw/Segmentation/CpeosTest/result/train_mask_results.tif'  

    test_img_path_2 = os.path.join(IMAGE_ROOT, 'GF2_test_image.tif')  
    output_path_2 = '/home/Dataset/nw/Segmentation/CpeosTest/result/GF2_test_image_results.tif'  


    # Load data  
    image, labels, nodata_value = load_data(train_img_path, label_img_path)  

    # Prepare dataset  
    X, y = load_dataset(X_path, y_path)  
    if X is None or y is None:  
        # Prepare dataset if not already saved  
        X, y = sample_dataset(image, labels, nodata_value, 50000, 11)  
        save_dataset(X, y, X_path, y_path) 

    # Split data  
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)  

    # Create datasets  
    train_dataset = RemoteSensingDataset(X_train, y_train)  
    val_dataset = RemoteSensingDataset(X_val, y_val)  

    # Create data loaders  
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  

    # # Initialize model  
    # num_classes = 10 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    # model = ResNet50(num_classes=num_classes).to(device)  

    # # Train the model  
    # train_model(model, train_loader, val_loader, model_path, num_epochs=8000)  

    # Classify an image  
    classify_image(model_path, test_img_path_1, output_path_1, nodata_value)
    classify_image(model_path, test_img_path_2, output_path_2, nodata_value)