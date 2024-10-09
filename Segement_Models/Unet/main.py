import torch  
import os
import numpy as np  
import torch.nn as nn  
import torch.optim as optim  
import torchvision.transforms as transforms  
from PIL import Image  
from torch.utils.data import DataLoader  
from unet import UNet  
from dataset import LargeImageDataset  

def train_model(model, train_loader, save_path, num_epochs=10, lr=1e-4):  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    # DataParallel for multiple GPUs  
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)  
    model.to(device)  

    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        for images, masks in train_loader:  
            images, masks = images.to(device), masks.to(device)  

            optimizer.zero_grad()  
            outputs = model(images)  
            loss = criterion(outputs, masks)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader):.4f}')  

    torch.save(model.state_dict(), save_path)  
    print('Model saved to {save_path}')  

def classify_image(image_path, model_path, patch_size=256, threshold=0.5):  
    """  
    Classify or segment an image using a trained U-Net model.  

    Parameters:  
    - image_path: str, path to the input image  
    - model_path: str, path to the trained model weights  
    - patch_size: int, size of the patches to process (default: 256)  
    - threshold: float, threshold for binary classification (default: 0.5)  

    Returns:  
    - segmented_image: np.ndarray, the segmented output image  
    """  
    # Load the trained model  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = UNet(in_channels=4, out_channels=1)  
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device)  
    model.eval()  

    # Load and preprocess the image  
    image = Image.open(image_path).convert('RGBA')  # 4通道影像  
    image = np.array(image)  
    h, w, _ = image.shape  

    # Prepare the output image  
    segmented_image = np.zeros((h, w), dtype=np.float32)  

    # Define transformation  
    transform = transforms.Compose([  
        transforms.ToTensor(),  
    ])  

    # Process the image in patches  
    with torch.no_grad():  
        for y in range(0, h, patch_size):  
            for x in range(0, w, patch_size):  
                # Extract patch  
                patch = image[y:y+patch_size, x:x+patch_size, :]  
                patch_h, patch_w, _ = patch.shape  

                # Pad patch if necessary  
                if patch_h < patch_size or patch_w < patch_size:  
                    patch = np.pad(patch, ((0, patch_size - patch_h), (0, patch_size - patch_w), (0, 0)), mode='constant')  

                # Transform and add batch dimension  
                patch = transform(patch).unsqueeze(0).to(device)  

                # Predict  
                output = model(patch)  
                output = torch.sigmoid(output).squeeze().cpu().numpy()  

                # Threshold the output  
                output = (output > threshold).astype(np.float32)  

                # Place the patch back into the output image  
                segmented_image[y:y+patch_h, x:x+patch_w] = output[:patch_h, :patch_w]  

    return segmented_image  

def main():  
    # 数据集路径  
    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    IMAGE_PATH = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    LABEL_PATH = os.path.join(IMAGE_ROOT, 'train_label.tif') 

    save_path = '/home/nw/Codes/Segement_Models/model_save/model_UNet.pth'
    test_img_path = os.path.join(IMAGE_ROOT, 'train_mask.tif')  
    output_path = '/home/Dataset/nw/Segmentation/CpeosTest/result/train_mask_results.tif'  
    
    # 数据转换  
    transform = transforms.Compose([  
        transforms.ToTensor(),  
    ])  

    # 数据加载器  
    dataset = LargeImageDataset(IMAGE_PATH, LABEL_PATH, patch_size=256, num_patches=5000, transform=transform)  
    train_loader = DataLoader(dataset, batch_size=192, num_workers=21, shuffle=True)  

    # 初始化模型  
    model = UNet(in_channels=4, out_channels=1)  

    # 训练模型  
    train_model(model, train_loader, save_path, num_epochs=500, lr=1e-4)  

    # 对新图像进行分类或分割  
    segmented_image = classify_image(test_img_path, save_path, patch_size=256, threshold=0.5)  

    # 保存分割结果  
    Image.fromarray((segmented_image * 255).astype(np.uint8)).save(output_path)  
    print(f'Segmented image saved to {output_path}')  

if __name__ == "__main__":  
    main()