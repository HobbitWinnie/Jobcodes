import os  
import glob  
import pandas as pd  
from PIL import Image  
from torch.utils.data import Dataset  
from torchvision import transforms  

class WHURS19DatasetLoader(Dataset):  
    def __init__(self, image_dir, csv_file, extensions=None, transform=None):  
        """  
        :param image_dir: 存储图片的目录。  
        :param csv_file: 包含图片标签信息的CSV文件路径。  
        :param extensions: 支持的文件扩展名列表，例如 ['.jpg', '.jpeg', '.png', '.bmp']。  
        :param transform: 图像预处理函数。  
        """  
        if extensions is None:  
            # 如果没有提供扩展名列表，使用默认扩展名  
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']  
        
        self.image_dir = image_dir  
        self.image_labels = pd.read_csv(csv_file)  
        self.extensions = extensions  
        self.transform = transform  

    def __len__(self):  
        return len(self.image_labels)  

    def __getitem__(self, idx):  
        # 从CSV中读取图像文件的基本名，不包括扩展名  
        img_basename = self.image_labels.iloc[idx, 0]  
        image_path = None  

        # 通过扩展名尝试匹配文件  
        for ext in self.extensions:  
            potential_path = os.path.join(self.image_dir, img_basename + ext)  
            if os.path.exists(potential_path):  
                image_path = potential_path  
                break  
        
        if image_path is None:  
            raise FileNotFoundError(f"No image found for {img_basename} with supported extensions {self.extensions}")  

        # 读取图片  
        try:  
            image = Image.open(image_path).convert('RGB')  
        except Exception as e:  
            print(f"Error loading image {image_path}: {e}")  
            return None, None  

        # 读取标签，从CSV的第二列开始  
        labels = self.image_labels.iloc[idx, 1:].astype('float32').values  

        # 应用图像预处理（如果提供了）  
        if self.transform:  
            image = self.transform(image)  

        return image, labels  
