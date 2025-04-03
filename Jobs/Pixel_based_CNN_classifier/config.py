
import torch  
from pathlib import Path

class ModelConfig:
    """模型训练配置参数"""
    
    def __init__(
        self,
        data_root: Path = Path("/home/Dataset/nw/Segmentation/CpeosTest"),
        model_save_path: Path = Path("/home/nw/Codes/Jobs/Pixel_based_CNN_classifier/model_save"),
        num_classes: int = 10,
        batch_size: int = 64,
        num_epochs: int = 500,
        learning_rate: float = 0.001,
        test_size: float = 0.5,
        sample_size: int = 50000,
        patch_size: int = 11
    ):
        # 路径配置
        self.train_image_path = data_root / "images/GF2_train_image.tif"
        self.label_image_path = data_root / "images/train_label.tif"
        self.sample_dir = data_root / "samples"
        self.model_path = model_save_path / "model_ResNet50_500epoch.pth"

        # 训练参数
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.test_size = test_size
        
        # 数据采样参数
        self.sample_size = sample_size
        self.patch_size = patch_size

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


class PredictConfig:  
    """预测任务配置参数"""  
    
    def __init__(  
        self,  
        model_path: Path = Path("/home/nw/Codes/Segement_Models/model_save/model_ResNet50_500epoch.pth"),  
        input_image: Path = Path("/home/Dataset/nw/Segmentation/CpeosTest/images/GF2_train_image.tif"),  
        output_path: Path = Path("/home/Dataset/nw/Segmentation/CpeosTest/result/prediction.tif"),  
        num_classes: int = 10,  
        patch_size: int = 7,  
        nodata_value: float = 15.0  
    ):  
        self.model_path = model_path  
        self.input_image = input_image  
        self.output_path = output_path  
        self.num_classes = num_classes  
        self.patch_size = patch_size  
        self.nodata_value = nodata_value  

    @property  
    def device(self) -> str:  
        return "cuda:0" if torch.cuda.is_available() else "cpu"  
