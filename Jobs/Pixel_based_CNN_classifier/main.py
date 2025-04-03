import os  
import logging  
import rasterio  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  

from nw.Codes.Jobs.Pixel_based_CNN_classifier.dataset import load_data, sample_dataset, load_dataset, save_dataset, RemoteSensingDataset  
from model_ResNet50 import ResNet50  
from model_CNN import SimpleCNN  

# 配置日志  
logging.basicConfig(level=logging.INFO)  

def prepare_data(train_img_path, label_img_path, X_path, y_path):  
    image, labels, nodata_value = load_data(train_img_path, label_img_path)  
    X, y = load_dataset(X_path, y_path)  
    if X is None or y is None:  
        X, y = sample_dataset(image, labels, nodata_value, 50000, 11)  
        save_dataset(X, y, X_path, y_path)  
    return X, y, nodata_value  

def train(model, train_dataset, val_dataset, save_path, num_epochs=10):  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    
    # 使用DataParallel包装模型  
    if torch.cuda.device_count() > 1:  
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")  
        model = nn.DataParallel(model)  
    model.to(device)  

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  

    train_loader = DataLoader(train_dataset, batch_size = 64* torch.cuda.device_count(),   
                            shuffle=True, num_workers=4)  
    val_loader = DataLoader(val_dataset, batch_size = 64* torch.cuda.device_count(),  
                           num_workers=4)  

    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  

        for inputs, labels in train_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  

        model.eval()  
        all_preds = []  
        all_labels = []  
        with torch.no_grad():  
            for inputs, labels in val_loader:  
                inputs, labels = inputs.to(device), labels.to(device)  
                outputs = model(inputs)  
                _, predicted = torch.max(outputs, 1)  
                all_preds.extend(predicted.cpu().numpy())  
                all_labels.extend(labels.cpu().numpy())  

        acc = accuracy_score(all_labels, all_preds)  
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc * 100:.2f}%')  

    torch.save(model.module.state_dict(), save_path)  # 保存原始模型  
    logging.info(f'Model saved to {save_path}')  

def classify_image(model_path, image_path, output_path, no_data_value, patch_size=7):  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    
    model = ResNet50(num_classes=10)  
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)  
    model.load_state_dict(torch.load(model_path))  
    model.to(device)  
    model.eval()  

    with rasterio.open(image_path) as src:  
        image = src.read()  
        profile = src.profile  

    pad_size = patch_size // 2  
    padded_image = np.pad(image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='edge')  
    h, w = image.shape[1], image.shape[2]  
    result_image = np.full((h, w), no_data_value, dtype=np.float32)  

    with torch.no_grad():  
        for row in tqdm(range(h)):  
            for col in range(w):  
                patch = padded_image[:, row:row + patch_size, col:col + patch_size]  
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  
                
                output = model(patch_tensor)  
                pred = output.argmax(dim=1).item()  
                result_image[row, col] = pred  

    profile.update(dtype=rasterio.float32, count=1)  
    with rasterio.open(output_path, 'w', **profile) as dst:  
        dst.write(result_image, 1)  

def main():  
    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    train_img_path = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    label_img_path = os.path.join(IMAGE_ROOT, 'train_label.tif')  

    SAMPLE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/samples'  
    X_path = os.path.join(SAMPLE_ROOT, 'X_sample_11_50000.npy')  
    y_path = os.path.join(SAMPLE_ROOT, 'Y_sample_11_50000.npy')  

    model_path = '/home/nw/Codes/Segement_Models/model_save/model_ResNet50_500epoch.pth'  
    output_path_1 = '/home/Dataset/nw/Segmentation/CpeosTest/result/train_mask_Res50_results.tif'  

    X, y, nodata_value = prepare_data(train_img_path, label_img_path, X_path, y_path)  
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)  

    train_dataset = RemoteSensingDataset(X_train, y_train)  
    val_dataset = RemoteSensingDataset(X_val, y_val)  

    num_classes = 10  
    model = ResNet50(num_classes=num_classes)  

    # 训练模型  
    train(model, train_dataset, val_dataset, model_path, num_epochs=500)  

    # 分类预测  
    classify_image(model_path, train_img_path, output_path_1, nodata_value)  

if __name__ == "__main__":  
    main()  