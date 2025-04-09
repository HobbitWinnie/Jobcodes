import torch  
import torch.nn as nn  
from torchvision import transforms  

class BaseMultiLabelClassifier(nn.Module):  
    """所有分类模型的基类"""  
    
    def __init__(self, num_classes, multi_gpu=False, device_ids=None):  
        super().__init__()  
        self.num_classes = num_classes  
        self.device_ids = device_ids or []
        self.multi_gpu = multi_gpu  

        self._validate_devices()    # 验证设备有效性  
        self._build_backbone()      # 构建模型  
        self._apply_multi_gpu()     # 配置多GPU  
        self._setup_optimizer()     # 初始化优化器 
    
    def _validate_devices(self):  
        """验证设备有效性"""  
        if self.multi_gpu and not torch.cuda.is_available():  
            raise RuntimeError("Requested multi-GPU but CUDA not available")  
            
        if any(idx >= torch.cuda.device_count() for idx in self.device_ids):  
            raise ValueError(  
                f"Device ids {self.device_ids} invalid for {torch.cuda.device_count()} GPUs"  
            ) 

    def _build_backbone(self):  
        """必须被子类实现的方法"""  
        raise NotImplementedError  
    
    def _apply_multi_gpu(self):  
        """统一处理多GPU并行"""  
        
        # 确定主设备  
        if self.device_ids:  
            self.main_device = f"cuda:{self.device_ids[0]}"  
        else:  
            self.main_device = "cuda:0" if torch.cuda.is_available() else "cpu"  
        
        # 移动模型到主设备  
        self.model = self.model.to(self.main_device) 

        # 启用DataParallel的条件判断优化  
        if self.multi_gpu and len(self.device_ids) > 1:  
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)  
            print(f"Enabled multi-GPU training on devices {self.device_ids}")   

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
        # 确保输入张量在正确设备  
        if not x.is_cuda and "cuda" in self.main_device:  
            x = x.to(self.main_device)  
        return self.model(x)  
    
    def __str__(self):  
        return f"{self.__class__.__name__}(main_device={self.main_device}, devices={self.device_ids}, classes={self.num_classes})"  
