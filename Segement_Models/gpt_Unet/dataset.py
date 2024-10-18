import numpy as np  
import torch  
from torch.utils.data import Dataset  
import random  
import torchvision.transforms as transforms  
from torchvision.transforms import functional as F  
from PIL import Image  


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
    reconstructed_image = np.zeros((H, W), dtype=np.int32)  
    patch_count = np.zeros((H, W), dtype=np.int32)  

    step = patch_size - overlap  
    patch_idx = 0  
    for i in range(0, H - patch_size + 1, step):  
        for j in range(0, W - patch_size + 1, step):  
            patch = patches[patch_idx]  
            for x in range(patch_size):  
                for y in range(patch_size):  
                    pixel_value = patch[x, y]  
                    # Update the reconstructed image with majority voting  
                    if patch_count[i + x, j + y] == 0:  
                        reconstructed_image[i + x, j + y] = pixel_value  
                    else:  
                        # If there's a tie, keep the existing value  
                        if np.count_nonzero(reconstructed_image[i + x, j + y] == pixel_value) > patch_count[i + x, j + y] / 2:  
                            reconstructed_image[i + x, j + y] = pixel_value  
                    patch_count[i + x, j + y] += 1  
            patch_idx += 1  

    return reconstructed_image  

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

        # Convert to PIL Image for augmentation  
        image_patch = Image.fromarray(image_patch)  
        label_patch = Image.fromarray(label_patch.squeeze(), 'L')  

        # Apply augmentation  
        if random.random() > 0.5:  
            image_patch = F.hflip(image_patch)  
            label_patch = F.hflip(label_patch)  

        if random.random() > 0.5:  
            image_patch = F.vflip(image_patch)  
            label_patch = F.vflip(label_patch)  

        angle = random.choice([0, 90, 180, 270])  
        image_patch = F.rotate(image_patch, angle)  
        label_patch = F.rotate(label_patch, angle)  

        # Convert back to tensor  
        image_patch = F.to_tensor(image_patch)  
        label_patch = torch.tensor(np.array(label_patch), dtype=torch.long)  

        # Generate mask_patch as a tensor with the same transform operations  
        mask_patch = (label_patch != self.labels_nodata).float()  

        return image_patch, label_patch, mask_patch  