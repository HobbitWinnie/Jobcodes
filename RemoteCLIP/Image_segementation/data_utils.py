import rasterio  
import os
import numpy as np  
import torch
from torch.utils.data import Dataset  

class SegmentationDataset(Dataset):  
    def __init__(self, image_path, label_path, patch_size=256, transform=None):  
        self.image_path = image_path  
        self.label_path = label_path  
        self.patch_size = patch_size  
        self.transform = transform  

        # Open the image and label files  
        with rasterio.open(self.image_path) as src:  
            self.image = src.read()  # Shape [C, H, W]  
            self.image_height, self.image_width = src.height, src.width  

        with rasterio.open(self.label_path) as src:  
            self.label = src.read(1)  # Assuming label is single channel  

        # Calculate the number of patches  
        self.num_patches_x = self.image_width // self.patch_size  
        self.num_patches_y = self.image_height // self.patch_size  

    def __len__(self):  
        return self.num_patches_x * self.num_patches_y  

    def __getitem__(self, idx):  
        # Calculate the x and y indices of the patch  
        x_idx = idx % self.num_patches_x  
        y_idx = idx // self.num_patches_x  

        # Calculate the pixel coordinates of the patch  
        x_start = x_idx * self.patch_size  
        y_start = y_idx * self.patch_size  

        # Extract the patch from the image and label  
        image_patch = self.image[:, y_start:y_start + self.patch_size, x_start:x_start + self.patch_size]  
        label_patch = self.label[y_start:y_start + self.patch_size, x_start:x_start + self.patch_size]  

        # Ensure data type consistency  
        image_patch = torch.tensor(image_patch, dtype=torch.float32)  
        label_patch = torch.tensor(label_patch, dtype=torch.long)  

        # Optionally, apply any transformations  
        if self.transform:  
            image_patch = self.transform(image_patch)  

        return image_patch, label_patch  

def load_data(image_path, label_path):  
    with rasterio.open(image_path) as src:  
        image = src.read()  # Shape [C, H, W]  
    with rasterio.open(label_path) as src:  
        labels = src.read(1)  # Shape [H, W]  
        nodata_value = src.nodata  

    return image, labels, nodata_value


def prepare_dataset(image, patch_size=256):  
    """  
    Split the image into non-overlapping patches of size patch_size x patch_size.  

    Parameters:  
    - image: np.ndarray, the image data  
    - patch_size: int, size of the patch to extract (default: 256)  

    Returns:  
    - X: np.ndarray, the array of patches  
    """  
    c, h, w = image.shape  
    X = []  

    # Ensure the image dimensions are at least patch_size in both dimensions  
    if h < patch_size or w < patch_size:  
        raise ValueError("Image dimensions should be larger than or equal to the patch size.")  

    # Iterate over the image with steps of patch_size  
    for row in range(0, h - patch_size + 1, patch_size):  
        for col in range(0, w - patch_size + 1, patch_size):  
            patch = image[:, row:row + patch_size, col:col + patch_size]  
            X.append(patch)  

    return np.array(X)  


def sample_dataset(image, labels, sample_size, patch_size=256):  
    """  
    Prepare dataset by sampling patches from the image.  

    Parameters:  
    - image: np.ndarray, the image data  
    - labels: np.ndarray, the label data  
    - sample_size: int, number of samples to extract  
    - patch_size: int, size of the patch to extract (default: 256)  

    Returns:  
    - X_sampled: np.ndarray, the sampled patches  
    - y_sampled: np.ndarray, the corresponding labels  
    """  
   
    c, h, w = image.shape  
    X, y = [], []  

     # Ensure the image dimensions are at least patch_size in both dimensions  
    if h < patch_size or w < patch_size:  
        raise ValueError("Image dimensions should be larger than or equal to the patch size.")  

 # Sample patches  
    for row in range(0, h - patch_size + 1, patch_size):  
        for col in range(0, w - patch_size + 1, patch_size):  
            patch = image[:, row:row + patch_size, col:col + patch_size]  
            patch_labels = labels[row:row + patch_size, col:col + patch_size]  

            # Use the center value of the patch as the label (or use major voting, etc.)  
            center_label = patch_labels[patch_size // 2, patch_size // 2]  
            X.append(patch)  
            y.append(center_label)  

    # Convert lists to arrays  
    X = np.array(X, dtype=np.float32)  
    y = np.array(y, dtype=np.float32)  

    # Sample a fraction or a fixed number of the data  
    indices = np.random.choice(range(len(X)), size=min(sample_size, len(X)), replace=False)  
    X_sampled = X[indices]  
    y_sampled = y[indices]  

    return X_sampled, y_sampled  


def save_dataset(X, y, X_path, y_path):  
        np.save(X_path, X)  
        np.save(y_path, y)  
        print(f"Dataset saved to {X_path} and {y_path}")  


def load_dataset(X_path, y_path):  
    if os.path.exists(X_path) and os.path.exists(y_path):  
        X = np.load(X_path)  
        y = np.load(y_path)  
        print(f"Dataset loaded from {X_path} and {y_path}")  
        return X, y  
    else:  
        print("Saved dataset not found. Please prepare the dataset first.")  
        return None, None  