import torch  
import os  
import numpy as np  
import torch.nn as nn  
import torch.optim as optim  
import torchvision.transforms as transforms  
import rasterio  
import logging  
from torch.utils.data import DataLoader  
from unet import UNet  
from dataset import LargeImageDataset  
import tqdm

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

def load_data(image_path, label_path=None):  
    logging.info(f"加载图像：{image_path}")  
    with rasterio.open(image_path) as src:  
        image = src.read()  # 形状 [C, H, W]  
        image_nodata = src.nodata  
        image_meta = src.meta  
        image_mask = (image[0] != image_nodata)  
        image = np.where(image_mask[np.newaxis, :, :], image,0)  

    if label_path:  
        logging.info(f"加载标签：{label_path}")  
        with rasterio.open(label_path) as src:  
            labels = src.read(1)  # 形状 [H, W]  
            nodata_value = src.nodata# 将nodata值替换为0  
            if nodata_value is not None:  
                labels = np.where(labels == nodata_value, 0, labels)
    else:  
        labels = None  

    return image, labels, image_mask, image_meta  

def train_model(model, train_loader, device, save_path, num_epochs=10, lr=1e-4):  
    model.to(device)  

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        for images, masks in train_loader:  
            images, masks = images.to(device), masks.to(device)  

            optimizer.zero_grad()  
            outputs = model(images)  

            loss = criterion(outputs, masks.long())  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  

        logging.info(f'Epoch {epoch+1}/{num_epochs},训练损失: {running_loss/len(train_loader):.4f}')  

    torch.save(model.state_dict(), save_path)  
    logging.info(f'模型已保存到 {save_path}')  

def classify_image(model, image, image_meta, model_path, output_path, device, patch_size=256, overlap=32):  
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.eval()  

    image = np.transpose(image, (1, 2, 0))  
    h, w, c = image.shape  
    segmented_image = np.zeros((h, w), dtype=np.int32)  
    confidence_map = np.zeros((h, w), dtype=np.float32)  

    transform = transforms.Compose([  
        transforms.ToTensor(),  
        transforms.ConvertImageDtype(torch.float32)  
    ])  

    with torch.no_grad():  
        for y in tqdm(range(0, h, patch_size - overlap), desc="处理行"):  
            for x in range(0, w, patch_size - overlap):  
                patch = image[y:min(y+patch_size, h), x:min(x+patch_size, w), :]  
                patch_h, patch_w, _ = patch.shape  

                if patch_h < patch_size or patch_w < patch_size:  
                    padding = ((0, max(0, patch_size - patch_h)), (0, max(0, patch_size - patch_w)), (0, 0))  
                    patch = np.pad(patch, padding, mode='reflect')  

                patch = transform(patch).unsqueeze(0).to(device)  

                output = model(patch)  
                probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()  
                labels = np.argmax(probabilities, axis=0)  
                max_probs = np.max(probabilities, axis=0)  

                # 更新分割图像和置信度图  
                for i in range(patch_h):  
                    for j in range(patch_w):  
                        y_pos, x_pos = y + i, x + j  
                        if y_pos < h and x_pos < w:  
                            if confidence_map[y_pos, x_pos] < max_probs[i, j]:  
                                segmented_image[y_pos, x_pos] = labels[i, j]  
                                confidence_map[y_pos, x_pos] = max_probs[i, j]  

    image_meta.update(dtype=rasterio.int32, count=1)  

    with rasterio.open(output_path, 'w', **image_meta) as dst:  
        dst.write(segmented_image, 1)  

    logging.info(f'分割图像已保存至 {output_path}')  

    # 保存置信度图  
    confidence_path = output_path.replace('.tif', '_confidence.tif')  
    confidence_meta = image_meta.copy()  
    confidence_meta.update(dtype=rasterio.float32, count=1)  
    with rasterio.open(confidence_path, 'w', **confidence_meta) as dst:  
        dst.write(confidence_map, 1)  

    logging.info(f'置信度图已保存至 {confidence_path}')  

    return segmented_image, confidence_map 

def main():  
    IMAGE_ROOT = '/home/Dataset/nw/Segmentation/CpeosTest/images'  
    IMAGE_PATH = os.path.join(IMAGE_ROOT, 'GF2_train_image.tif')  
    LABEL_PATH = os.path.join(IMAGE_ROOT, 'train_label.tif')  

    save_path = '/home/nw/Codes/Segement_Models/model_save/model_UNet.pth'  
    test_img_path = os.path.join(IMAGE_ROOT, 'train_mask.tif')  
    output_path = '/home/Dataset/nw/Segmentation/CpeosTest/result/train_mask_Unet_results.tif'  

    PATCH_SIZE = 256  
    PATCH_NUMBER = 10000  
    BATCH_SIZE = 128  
    EPOCHS = 1000  
    LEARNING_RATE = 0.001  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logging.info(f"使用设备: {device}")  

    image, labels, image_mask, _ = load_data(IMAGE_PATH, LABEL_PATH)  

    dataset = LargeImageDataset(image, labels, patch_size=PATCH_SIZE, num_patches=PATCH_NUMBER)  
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)  

    logging.info("初始化UNet模型")  
    num_classes = 10  
    model = UNet(in_channels=4, out_channels=num_classes)  
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)  
    model.to(device)  

    # 训练模型  
    train_model(model, train_loader, device, save_path, num_epochs=EPOCHS, lr=LEARNING_RATE)  

    # 对新图像进行分类  
    test_image, _, _, test_image_meta = load_data(test_img_path)  
    classify_image(model, test_image, test_image_meta, save_path, output_path, device, patch_size=PATCH_SIZE)  

if __name__ == "__main__":  
    main()