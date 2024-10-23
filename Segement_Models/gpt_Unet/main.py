import os  
import torch  
import rasterio  
import torch.nn as nn  
import logging  
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from torch.utils.data import DataLoader  
from torch.cuda.amp import GradScaler, autocast  
from tqdm import tqdm  
from torch.nn import functional as F

from dataset import RemoteSensingDataset, reconstruct_image_from_patches, split_image_into_patches  
from utils import load_data, mean_iou, multiclass_dice_coefficient
from model import UNet  

# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  


def train(model, train_loader, val_loader, device, epochs, learning_rate, save_path):  
    model.to(device)  
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    scaler = GradScaler()  
    scheduler = ReduceLROnPlateau(optimizer,'max', patience=5, factor=0.5)  

    best_val_dice = 0  
    patience, patience_counter = 100, 0  

    for epoch in range(epochs):  
        model.train()  
        total_loss = total_dice = total_iou = 0  

        for batch in train_loader:  
            img_patch, mask_patch = [b.to(device) for b in batch]  
            optimizer.zero_grad()  

            with autocast():  
                outputs = model(img_patch)  
                loss = criterion(outputs, mask_patch.long())  

            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  

            total_loss += loss.item()  
            pred = F.softmax(outputs, dim=1)  
            total_dice += multiclass_dice_coefficient(pred, mask_patch).item()  
            total_iou += mean_iou(pred, mask_patch).item()  

        average_loss = total_loss / len(train_loader)  
        average_dice = total_dice / len(train_loader)  
        average_iou = total_iou / len(train_loader)  

        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)  

        logging.info(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {average_loss:.4f}, "  
                     f"Train Dice: {average_dice:.4f}, Train IoU: {average_iou:.4f}, "  
                     f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")  

        scheduler.step(val_dice)  

        if val_dice > best_val_dice:  
            best_val_dice = val_dice  
            torch.save(model.state_dict(), save_path)  
            logging.info(f"Model saved to {save_path}")  
            patience_counter = 0  
        else:  
            patience_counter += 1  

        if patience_counter >= patience:  
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")  
            break  

def validate(model, val_loader, criterion, device):  
    model.eval()  
    total_loss = 0  
    total_dice = 0  
    total_iou = 0  
    
    with torch.no_grad():  
        for batch in val_loader:  
            img_patch, mask_patch = [b.to(device) for b in batch]  

            with autocast():  
                outputs = model(img_patch)  
                loss = criterion(outputs, mask_patch.long())  
                
            total_loss += loss.item()  
            pred = F.softmax(outputs, dim=1)  
            total_dice += multiclass_dice_coefficient(pred, mask_patch).item()  
            total_iou += mean_iou(pred, mask_patch).item()  

    average_loss = total_loss / len(val_loader)  
    average_dice = total_dice / len(val_loader)  
    average_iou = total_iou / len(val_loader)  
    return average_loss, average_dice, average_iou  


def predict(model, save_path, test_image_paths, output_paths, patch_size, overlap, device):  
    model.load_state_dict(torch.load(save_path, map_location=device))  
    model.eval()  

    for test_image_path, output_path in zip(test_image_paths, output_paths):  
        logging.info(f"Predicting for {test_image_path}")  
        test_image, _, image_profile = load_data(test_image_path)# image, labels, image_meta  
        patches = split_image_into_patches(test_image, patch_size, overlap)  
        predictions = []  

        with torch.no_grad():  
            for patch in tqdm(patches, desc="Processing patches"):  
                patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  
                output = model(patch)  
                pred = F.softmax(output, dim=1).squeeze().cpu().numpy()  
                predictions.append(pred)  

        reconstructed_prediction = reconstruct_image_from_patches(predictions, test_image.shape, patch_size, overlap)  

        #更新图像配置  
        image_profile.update(dtype=rasterio.uint8, count=1, nodata=0)  

        with rasterio.open(output_path, 'w', **image_profile) as dst:  
            # 写入最终预测  
            dst.write(reconstructed_prediction.astype(rasterio.uint8), 1)  

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
    PATCH_NUMBER = 5000  
    OVERLAP = 64  
    BATCH_SIZE = 128  
    EPOCHS = 1000  
    LEARNING_RATE = 0.001  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    logging.info("Initializing UNet model")  
    model = UNet(in_channels=4, out_channels=10, dropout_rate=0.1)  
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)  
    model.to(device)  

    try:  
        image, labels, _ = load_data(IMAGE_PATH, LABEL_PATH)  
    except Exception as e:  
        logging.error(f"Error loading data: {e}")  
        return  

    dataset = RemoteSensingDataset(image, labels, patch_size=PATCH_SIZE, num_patches=PATCH_NUMBER)  
    train_size = int(0.8 * len(dataset))  
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])  

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)  
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)  

    train(model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE, save_path)  

    predict(model, save_path, test_img_paths, output_paths, PATCH_SIZE, OVERLAP, device)  

if __name__ == "__main__":  
    main()