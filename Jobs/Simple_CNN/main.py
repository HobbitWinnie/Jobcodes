import os  
import logging  
import rasterio  
import torch  
import torch.multiprocessing as mp  
import torch.distributed as dist  
import torch.optim as optim  
import torch.nn as nn  
import numpy as np  
from torch.utils.data import DataLoader, DistributedSampler  
from sklearn.model_selection import train_test_split  
from tqdm import tqdm  
from sklearn.metrics import accuracy_score 

from data_utils import load_data, sample_dataset, load_dataset, save_dataset, RemoteSensingDataset  
from model_ResNet50 import ResNet50  
from model_CNN import SimpleCNN

# 配置日志  
logging.basicConfig(level=logging.INFO)  

# 初始化分布式环境  
def setup(rank, world_size):  
    os.environ['MASTER_ADDR'] = '127.0.0.1'  
    os.environ['MASTER_PORT'] = '12355'  
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  

def cleanup():  
    dist.destroy_process_group()  

# 数据准备  
def prepare_data(train_img_path, label_img_path, X_path, y_path):  
    image, labels, nodata_value = load_data(train_img_path, label_img_path)  
    X, y = load_dataset(X_path, y_path)  
    if X is None or y is None:  
        X, y = sample_dataset(image, labels, nodata_value, 50000, 11)  
        save_dataset(X, y, X_path, y_path)  
    return X, y, nodata_value  

# 模型训练  
def train(rank, world_size, model, train_dataset, val_dataset, save_path, num_epochs=10):  
    setup(rank, world_size)  
    device = torch.device(f'cuda:{rank}')  
    model.to(device)  
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])  

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)  
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)  

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=4)  
    val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler, num_workers=4)  

    for epoch in range(num_epochs):  
        model.train()  
        train_sampler.set_epoch(epoch)  
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
        
        if rank == 0:  
            logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {acc * 100:.2f}%')  

    if rank == 0:  
        torch.save(model.state_dict(), save_path)  
        logging.info(f'Model saved to {save_path}')  
    
    cleanup()  

# 图像分类  
def classify_image(rank, world_size, model_path, image_path, output_path, no_data_value, patch_size=7):  
    setup(rank, world_size)  
    device = torch.device(f'cuda:{rank}')  

    model = ResNet50(num_classes=10).to(device)  
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])  
    model.eval()  

    with rasterio.open(image_path) as src:  
        image = src.read()  
        profile = src.profile  

    pad_size = patch_size // 2  
    padded_image = np.pad(image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='edge')  
    h, w = image.shape[1], image.shape[2]  
    result_image = np.full((h, w), no_data_value, dtype=np.float32)  

    with torch.no_grad():  
        for row in tqdm(range(rank, h, world_size), desc=f"Processing rows (GPU {rank})"):  
            for col in range(w):  
                patch = padded_image[:, row:row + patch_size, col:col + patch_size]  
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  

                output = model(patch_tensor)  
                pred = output.argmax(dim=1).item()  

                result_image[row, col] = pred  

    if rank == 0:  
        profile.update(dtype=rasterio.float32, count=1)  
        with rasterio.open(output_path, 'w', **profile) as dst:  
            dst.write(result_image, 1)  

    cleanup()  

# 主函数  
def main():  
    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    train_img_path = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    label_img_path = os.path.join(IMAGE_ROOT, 'train_label.tif')  

    SAMPLE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/samples'  
    X_path = os.path.join(SAMPLE_ROOT, 'X_sample_11_50000.npy')  
    y_path = os.path.join(SAMPLE_ROOT, 'Y_sample_11_50000.npy')  

    model_path = '/home/nw/Codes/Segement_Models/model_save/model_SimpleCNN_5000epoch.pth'  

    test_img_path_1 = os.path.join(IMAGE_ROOT, 'train_mask.tif')  
    output_path_1 = '/home/Dataset/nw/Segmentation/CpeosTest/result/train_mask_Res50_results.tif'  

    # 数据准备  
    X, y, nodata_value = prepare_data(train_img_path, label_img_path, X_path, y_path)  

    # 数据集拆分  
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)  

    # 创建数据集  
    train_dataset = RemoteSensingDataset(X_train, y_train)  
    val_dataset = RemoteSensingDataset(X_val, y_val)  

    num_classes = 10  
    model = SimpleCNN(num_classes=num_classes)  

    # 启动分布式训练  
    world_size = torch.cuda.device_count()  
    mp.spawn(train, args=(world_size, model, train_dataset, val_dataset, model_path, 500), nprocs=world_size, join=True)  

    # 使用分布式推理  
    mp.spawn(classify_image, args=(world_size, model_path, test_img_path_1, output_path_1, nodata_value), nprocs=world_size, join=True)  

if __name__ == "__main__":  
    main()