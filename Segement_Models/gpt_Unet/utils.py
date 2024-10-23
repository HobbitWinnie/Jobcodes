import torch  
import rasterio  
import numpy as np  
import logging  


def load_data(image_path, label_path=None):  
    logging.info(f"Loading image from {image_path}")  
    with rasterio.open(image_path) as src:  
        image = src.read()  
        image_nodata = int(src.nodata)  
        image_meta = src.meta  

    image_mask = (image[0] != image_nodata)  
    image = np.where(image_mask, image, 0)  

    if label_path:  
        logging.info(f"Loading labels from {label_path}")  
        with rasterio.open(label_path) as src:  
            labels = src.read(1)  
            labels_nodata = int(src.nodata)  
        label_mask = (labels != labels_nodata)  
        labels = np.where(label_mask, labels, 0)  
    else:  
        labels = None  

    return image, labels, image_meta  

def multiclass_dice_coefficient(pred, target, smooth=1e-5):  
    num_classes = pred.shape[1]  
    dice_scores = []  
    for class_idx in range(num_classes):  
        class_pred = pred[:, class_idx,...]  
        class_target = (target == class_idx).float()  
        intersection = (class_pred * class_target).sum()  
        union = class_pred.sum() + class_target.sum()  
        dice_scores.append((2. * intersection + smooth) / (union + smooth))  
    return torch.stack(dice_scores).mean()  

def mean_iou(pred, target):  
    num_classes = pred.shape[1]  
    iou_scores = []  
    for class_idx in range(num_classes):  
        class_pred = pred[:, class_idx, ...]  
        class_target = (target == class_idx).float()  
        intersection = (class_pred * class_target).sum()  
        union = class_pred.sum() + class_target.sum() - intersection  
        iou_scores.append((intersection + 1e-5) / (union + 1e-5))  
    return torch.stack(iou_scores).mean()  