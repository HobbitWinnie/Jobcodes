import numpy as np  
import torch  
from torch.utils.data import Dataset  
import random  

def split_image_into_patches(image, patch_size=256, overlap=128):  
    patches = []  
    C, H, W = image.shape  
    step = patch_size - overlap  
    for i in range(0, H - patch_size + 1, step):  
        for j in range(0, W - patch_size + 1, step):  
            patch = image[:, i:i + patch_size, j:j + patch_size]  
            patches.append(patch)  
    return patches  

def reconstruct_image_from_patches(patches, image_shape, patch_size=256, overlap=128):  
    C, H, W = image_shape  
    reconstructed_image = np.zeros((C, H, W), dtype=np.float32)  
    patch_count = np.zeros((C, H, W), dtype=np.float32)  

    step = patch_size - overlap  
    patch_idx = 0  
    for i in range(0, H - patch_size + 1, step):  
        for j in range(0, W - patch_size + 1, step):  
            reconstructed_image[:, i:i + patch_size, j:j + patch_size] += patches[patch_idx]  
            patch_count[:, i:i + patch_size, j:j + patch_size] += 1  
            patch_idx += 1  

    patch_count = np.where(patch_count == 0, 1, patch_count)  
    return reconstructed_image / patch_count  

class RemoteSensingDataset(Dataset):  
    def __init__(self, image, labels, labels_nodata=0, patch_size=256, num_patches=1000):  
        self.image = image  
        self.labels = labels  
        self.labels_nodata = labels_nodata  
        self.patch_size = patch_size  
        self.num_patches = num_patches  

        # Ensure the image and labels have the correct dimensions  
        self.image = np.transpose(self.image, (1, 2, 0))  # [H, W, C]  
        self.labels = np.expand_dims(self.labels, axis=0)  # [1, H, W]  

        self.h, self.w, _ = self.image.shape  

    def __len__(self):  
        return self.num_patches  

    def __getitem__(self, idx):  
        # Ensure patch size is not larger than the image dimensions  
        if self.h < self.patch_size or self.w < self.patch_size:  
            raise ValueError(f"Patch size {self.patch_size} is too large for image size {self.h}x{self.w}")  

        # Randomly select the starting position of the patch  
        max_x = self.w - self.patch_size  
        max_y = self.h - self.patch_size  
        x = random.randint(0, max_x)  
        y = random.randint(0, max_y)  

        # Extract the patch  
        image_patch = self.image[y:y+self.patch_size, x:x+self.patch_size, :]  
        label_patch = self.labels[:, y:y+self.patch_size, x:x+self.patch_size]  

        # Generate mask based on labels  
        mask_patch = (label_patch != self.labels_nodata).astype(np.float32)  

        # Convert to tensors  
        image_patch = torch.tensor(np.transpose(image_patch, (2, 0, 1)), dtype=torch.float32)  # [C, H, W]  
        label_patch = torch.tensor(label_patch.squeeze(), dtype=torch.long)  # [H, W]  
        mask_patch = torch.tensor(mask_patch.squeeze(), dtype=torch.float32)  # [H, W]  

        return image_patch, label_patch, mask_patch  