import torch  
import torch.nn as nn  
import open_clip  
import traceback  

class BaseRemoteCLIPSeg(nn.Module):  
    def __init__(self, model_name, num_classes, input_size=224, ckpt_path=None, freeze_clip=True, in_channels=4):  
        super().__init__()  
        self.model_name = model_name  
        self.num_classes = num_classes  
        self.input_size = input_size  
        self.in_channels = in_channels  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self._init_clip_model(ckpt_path, freeze_clip)  
        self.to(self.device)  

    def _init_clip_model(self, ckpt_path=None, freeze_clip=True):  
        try:  
            model, _, _ = open_clip.create_model_and_transforms(self.model_name)  
            assert (model is not None) and (model.visual is not None), "CLIP模型加 载失败"  
            self.visual_encoder = model.visual.to(self.device)  
            self.visual_encoder.eval()  

            # 输入层适配in_channels通道  
            original_conv1 = self.visual_encoder.conv1  
            self.visual_encoder.conv1 = nn.Conv2d(  
                self.in_channels,  
                original_conv1.out_channels,  
                kernel_size=original_conv1.kernel_size,  
                stride=original_conv1.stride,  
                padding=original_conv1.padding,  
                bias=(original_conv1.bias is not None)  
            ).to(self.device)  
            
            with torch.no_grad():  
                # 复制前三通道，初始化第4个通道  
                self.visual_encoder.conv1.weight[:, :3, :, :] = original_conv1.weight  
                mean_weight = original_conv1.weight[:, :3, :, :].mean(dim=1, keepdim=True)  
                self.visual_encoder.conv1.weight[:, 3:4, :, :] = mean_weight  

            if freeze_clip:  
                for param in self.visual_encoder.parameters():  
                    param.requires_grad = False  

            if ckpt_path:  
                ckpt = torch.load(ckpt_path, map_location=self.device)  
                if isinstance(ckpt, dict) and 'state_dict' in ckpt:  
                    ckpt = ckpt['state_dict']  
                self.load_state_dict(ckpt, strict=False)  
        
        except Exception as e:  
            print(f"CLIP模型加载失败: {e}")  
            traceback.print_exc()  
            raise  

    def _validate_input(self, x):  
        if x.dim() != 4:  
            raise ValueError(f"输入应为4维张量，实际为{ x.dim() }维")  
        if x.shape[1] != self.in_channels:  
            raise ValueError(f"期望 {self.in_channels} 通道，实际为{ x.shape[1] }通道")  
        if x.shape[2:] != (self.input_size, self.input_size):  
            raise ValueError(f"期望输入尺寸为{self.input_size}x{self.input_size}，实际为{ x.shape[2] }x{ x.shape[3] }")  

    def forward(self, x, *args, **kwargs):  
        raise NotImplementedError("子类需实现forward方法")  