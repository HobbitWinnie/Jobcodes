from torch.utils.data import Dataset  
import numpy as np  
import random  
import torch  


class LargeImageDataset(Dataset):  
    def __init__(self, image, label, patch_size=256, num_patches=1000):  
        # Ensure the image is in [H, W, C] format for easier patch extraction  
        self.image = np.transpose(image, (1, 2, 0))  
        self.label = np.array(label)  
        self.patch_size = patch_size  
        self.num_patches = num_patches  

        self.h, self.w, _ = self.image.shape  

    def __len__(self):  
        return self.num_patches  

    def __getitem__(self, idx):  
        # Ensure patch size is not larger than the image dimensions  
        if self.h < self.patch_size or self.w < self.patch_size:  
            raise ValueError(f"Patch size {self.patch_size} is too large for image of size {self.h}x{self.w}")  

        # Randomly select the starting position of the patch  
        max_x = self.w - self.patch_size  
        max_y = self.h - self.patch_size  
        x = random.randint(0, max_x)  
        y = random.randint(0, max_y)  

        # Extract the patch  
        image_patch = self.image[y:y+self.patch_size, x:x+self.patch_size, :]  
        label_patch = self.label[y:y+self.patch_size, x:x+self.patch_size]  

        # Ensure the type is float32  
        image_patch = image_patch.astype(np.float32)  
        label_patch = label_patch.astype(np.float32)  

        # Apply transformations if any  
        if self.transform:  
            image_patch = self.transform(image_patch)  

        # Convert labels to tensor  
        label_patch = torch.tensor(label_patch, dtype=torch.float32)  
        
        return image_patch, label_patch