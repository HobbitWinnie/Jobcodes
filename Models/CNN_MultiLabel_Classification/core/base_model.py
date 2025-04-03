import torch  
import torch.nn as nn  
from torchvision import transforms  

class BaseMultiLabelClassifier(nn.Module):  
    """所有分类模型的基类"""  
    
    def __init__(self, num_classes, multi_gpu=True, device=None):  
        super().__init__()  
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  
        self.num_classes = num_classes  
        self.multi_gpu = multi_gpu  
        self._build_backbone()  
        self._apply_multi_gpu()  
        self._setup_optimizer()  
    
    def _build_backbone(self):  
        """必须被子类实现的方法"""  
        raise NotImplementedError  
    
    def _apply_multi_gpu(self):  
        """统一处理多GPU并行"""  
        if self.multi_gpu and torch.cuda.device_count() > 1:  
            self.model = nn.DataParallel(self.model)  
        self.model = self.model.to(self.device)  

    def _setup_optimizer(self):  
        """配置默认优化器"""  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  
            self.optimizer, mode='min', factor=0.1, patience=5  
        )  
    
    @property  
    def input_size(self):  
        """模型预期输入尺寸"""  
        return (224, 224)  # 默认尺寸，可被子类覆盖  
        
    @property  
    def preprocess(self):  
        """通用预处理流程"""  
        return transforms.Compose([  
            transforms.Resize(self.input_size),  
            transforms.ToTensor(),  
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ])  
    
    def forward(self, x):  
        return self.model(x)  

    def __str__(self):  
        return f"{self.__class__.__name__}(device={self.device}, classes={self.num_classes})"  