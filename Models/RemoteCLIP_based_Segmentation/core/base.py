import torch  
import torch.nn as nn  
import open_clip  
import traceback  
import logging
from pathlib import Path


class BaseRemoteCLIPSeg(nn.Module):  
    def __init__(
            self, 
            model_name, 
            num_classes, 
            input_size=224, 
            ckpt_path=None, 
            freeze_clip=True, 
            in_channels=4,
            device_ids: list = None,  
    ):  
        super().__init__()  
        self.model_name = model_name  
        self.num_classes = num_classes  
        self.input_size = input_size  
        self.in_channels = in_channels  
        self.device_ids = device_ids or []  
        
        self.logger = logging.getLogger(self.__class__.__name__)  
        
        self.main_device = self._determine_main_device()  
        self._validate_devices()  
        self.logger.info(f"Main device: {self.main_device}")  
        self._init_clip_model(ckpt_path, freeze_clip)  
        self.to(self.main_device)  
    
    def _validate_devices(self):  
        if self.device_ids:  
            if not torch.cuda.is_available():  
                raise RuntimeError("指定了 device_ids 但 CUDA 不可用")  
            available = list(range(torch.cuda.device_count()))  
            invalid = [i for i in self.device_ids if i not in available]  
            if invalid:  
                raise ValueError(f"无效 device_ids {invalid}, 可选范围: {available}")  

    def _determine_main_device(self):  
        if self.device_ids:  
            return f"cuda:{self.device_ids[0]}"  
        return "cuda:0" if torch.cuda.is_available() else "cpu"  
    
    def _init_clip_model(self, model_name, ckpt_path=None, freeze_clip=True):  
        try:  
            model, _, _ = open_clip.create_model_and_transforms(model_name)  
            assert (model is not None) and (model.visual is not None), "CLIP模型加 载失败"  
            visual_encoder = model.visual.to(self.main_device)  
            visual_encoder.eval()  

            # in_channels 适配  
            self._adjust_conv1(visual_encoder, self.in_channels)  
 
            if freeze_clip:  
                for param in self.visual_encoder.parameters():  
                    param.requires_grad = False  

            if ckpt_path is not None:  
                ckpt_path = Path(ckpt_path)  
                if not ckpt_path.is_file():  
                    raise FileNotFoundError(f"ckpt_path文件不存在: {ckpt_path}")  
                self.logger.info(f"加载CLIP权重: {ckpt_path}")  

                ckpt = torch.load(str(ckpt_path), map_location=self.main_device)  
                if isinstance(ckpt, dict) and "state_dict" in ckpt:  
                    ckpt = ckpt["state_dict"]  
                missing, unexpected = visual_encoder.load_state_dict(ckpt, strict=False)  
                self.logger.info(f"load_state_dict missing={missing}, unexpected={unexpected}")  

            # 多GPU包装  
            if self.device_ids and len(self.device_ids) > 1:  
                visual_encoder = nn.DataParallel(visual_encoder, device_ids=self.device_ids)  
                self.logger.info(f"使用多GPU: {self.device_ids}")  
            else:  
                self.logger.info("使用单GPU或CPU")  

            self.visual_encoder = visual_encoder  
            self.logger.info("CLIP视觉主干初始化完成。")  

        except Exception as e:  
            print(f"CLIP模型加载失败: {e}")  
            traceback.print_exc()  
            raise  

    @staticmethod  
    def _adjust_conv1(visual_encoder, in_channels):  
        original_conv1 = visual_encoder.conv1  
        new_conv1 = nn.Conv2d(  
            in_channels,  
            original_conv1.out_channels,  
            kernel_size=original_conv1.kernel_size,  
            stride=original_conv1.stride,  
            padding=original_conv1.padding,  
            bias=(original_conv1.bias is not None),  
        ).to(original_conv1.weight.device)  

        # 复制前三通道，初始化第4个通道  
        with torch.no_grad():  
            new_conv1.weight[:, :3, :, :] = original_conv1.weight  
            if in_channels > 3:  
                mean_weight = original_conv1.weight[:, :3, :, :].mean(dim=1, keepdim=True)  
                for i in range(3, in_channels):  
                    new_conv1.weight[:, i:i+1, :, :] = mean_weight  
        visual_encoder.conv1 = new_conv1  

    def _validate_input(self, x):  
        if x.dim() != 4:  
            raise ValueError(f"输入应为4维张量，实际为{ x.dim() }维")  
        if x.shape[1] != self.in_channels:  
            raise ValueError(f"期望 {self.in_channels} 通道，实际为{ x.shape[1] }通道")  
        if x.shape[2:] != (self.input_size, self.input_size):  
            raise ValueError(f"期望输入尺寸为{self.input_size}x{self.input_size}，实际为{ x.shape[2] }x{ x.shape[3] }")  

    def forward(self, x, *args, **kwargs):  
        raise NotImplementedError("子类需实现forward方法")  
    
    @property  
    def encoder(self):  
        return self.visual_encoder.module if hasattr(self.visual_encoder, "module") else self.visual_encoder  
