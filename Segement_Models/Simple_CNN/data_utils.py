import rasterio  
import os
import numpy as np  
from torch.utils.data import Dataset  

class RemoteSensingDataset(Dataset):  
    def __init__(self, images, labels, transform=None):  
        self.images = images  
        self.labels = labels  
        self.transform = transform  

    def __len__(self):  
        return len(self.images)  

    def __getitem__(self, idx):  
        sample = self.images[idx]  
        label = self.labels[idx]  
        if self.transform:  
            sample = self.transform(sample)  
        return sample, label
    
def load_data(image_path, label_path):  
    with rasterio.open(image_path) as src:  
        image = src.read()  # Shape [C, H, W]  
    with rasterio.open(label_path) as src:  
        labels = src.read(1)  # Shape [H, W]  
        nodata_value = src.nodata  

    return image, labels, nodata_value

def prepare_dataset(image, patch_size=7):  
    c, h, w = image.shape  
    pad_size = patch_size // 2  
    padded_image = np.pad(image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='edge')  
    X=[]  
    
    for row in range(h):  
        for col in range(w):  
            patch = padded_image[:, row:row + patch_size, col:col + patch_size]  
            X.append(patch)  
    
    return np.array(X)


def sample_dataset(image, labels, no_data_value, sample_size, patch_size=7,):  
    """  
    Prepare dataset by sampling patches from the image and avoiding nodata values.  

    Parameters:  
    - image: np.ndarray, the image data  
    - labels: np.ndarray, the label data  
    - patch_size: int, size of the patch to extract (default: 7)  
    - no_data_value: the value in labels that represents no data (default: -1)  
    - sample_fraction: float, fraction of the total data to sample (default: 0.1)  

    Returns:  
    - X: np.ndarray, the sampled patches  
    - y: np.ndarray, the corresponding labels  
    """  
    c, h, w = image.shape  
    pad_size = patch_size // 2  
    padded_image = np.pad(image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='edge')  
    X, y = [], []  

    for row in range(h):  
        for col in range(w):  
            if labels[row, col] == no_data_value:  
                continue  # Skip patches with nodata label  
            
            patch = padded_image[:, row:row + patch_size, col:col + patch_size]  
            label = labels[row, col]  
            X.append(patch)  
            y.append(label)  
    
    # Convert lists to arrays  
    X = np.array(X, dtype=np.float32)  
    y = np.array(y, dtype=np.float32)  
    
    # Sample a fraction of the data  
    indices = np.random.choice(range(len(X)), size=sample_size, replace=False)  
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