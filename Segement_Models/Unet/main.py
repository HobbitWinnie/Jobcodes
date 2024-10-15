import torch  
import os  
import numpy as np  
import torch.nn as nn  
import torch.optim as optim  
import torchvision.transforms as transforms  
import rasterio  
from torch.utils.data import DataLoader  
from unet import UNet   
from dataset import LargeImageDataset  

def train_model(model, train_loader, save_path, num_epochs=10, lr=1e-4):  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    # Use DataParallel for multiple GPUs  
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)  
    model.to(device)  

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        for images, masks in train_loader:  
            # Ensure images and masks are on the correct device  
            images, masks = images.to(device), masks.to(device)  

            optimizer.zero_grad()  
            outputs = model(images)  

            # Ensure masks are of type long for CrossEntropyLoss  
            loss = criterion(outputs, masks.long())  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader):.4f}')  

    torch.save(model.state_dict(), save_path)  
    print(f'Model saved to {save_path}')  

def classify_image(image_path, model_path, output_path, num_classes, patch_size=256):  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    # Initialize model with num_classes  
    model = UNet(in_channels=4, out_channels=num_classes)  
    
    state_dict = torch.load(model_path, map_location=device)  
    
    # Manage state dict for DataParallel  
    if any(k.startswith('module.') for k in state_dict.keys()):  
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  
        
    model.load_state_dict(state_dict)  

    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)  

    model.to(device)  
    model.eval()  

    with rasterio.open(image_path) as src:  
        image = src.read()  # Image shape: [C, H, W]  
        meta = src.meta  

    # Transpose image to shape [H, W, C] for processing  
    image = np.transpose(image, (1, 2, 0))  
    h, w, _ = image.shape  

    # Initialize output image  
    segmented_image = np.zeros((h, w), dtype=np.float32)  

    transform = transforms.Compose([  
        transforms.ToTensor(),  
        transforms.ConvertImageDtype(torch.float32)  # Ensure float32 for model input  
    ])  

    with torch.no_grad():  
        for y in range(0, h, patch_size):  
            for x in range(0, w, patch_size):  
                # Extract patch  
                patch = image[y:min(y+patch_size, h), x:min(x+patch_size, w), :]  
                patch_h, patch_w, _ = patch.shape  

                # Pad patch if necessary  
                if patch_h < patch_size or patch_w < patch_size:  
                    padding = ((0, patch_size - patch_h), (0, patch_size - patch_w), (0, 0))  
                    patch = np.pad(patch, padding, mode='constant')  

                # Transform and add batch dimension  
                patch = transform(patch).unsqueeze(0).to(device)  

                # Predict  
                output = model(patch)  
                output = torch.sigmoid(output).squeeze().cpu().numpy()  

                # Convert multi-class output to single class per pixel  
                output = np.argmax(output, axis=0).astype(np.int32) 

                # Place the patch back into the output image, limited by original patch size  
                segmented_image[y:y+patch_h, x:x+patch_w] = output[:patch_h, :patch_w]  

    # Update metadata for saving  
    meta.update(dtype=rasterio.float32, count=1)  

    # Save the segmented image  
    with rasterio.open(output_path, 'w', **meta) as dst:  
        dst.write(segmented_image, 1)  

    print(f'Segmented image saved to {output_path}')  
    return segmented_image  

def main():  
    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    IMAGE_PATH = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    LABEL_PATH = os.path.join(IMAGE_ROOT, 'train_label.tif')  

    save_path = '/home/nw/Codes/Segement_Models/model_save/model_UNet.pth'  
    test_img_path = os.path.join(IMAGE_ROOT, 'train_mask.tif')  
    output_path = '/home/Dataset/nw/Segmentation/CpeosTest/result/train_mask_Unet_results.tif'  
    
    transform = transforms.Compose([  
        transforms.ToTensor(),  
    ])  

    patch_size=256
    # dataset = LargeImageDataset(IMAGE_PATH, LABEL_PATH, patch_size=patch_size, num_patches=5000, transform=transform)  
    # train_loader = DataLoader(dataset, batch_size=192, num_workers=4, shuffle=True)  

    num_classes = 10  
    # model = UNet(in_channels=4, out_channels=num_classes)  

    # Train model  
    # train_model(model, train_loader, save_path, num_epochs=10, lr=1e-4)  

    # Classify a new image  
    classify_image(test_img_path, save_path, output_path, num_classes=num_classes, patch_size=patch_size)  

if __name__ == "__main__":  
    main()