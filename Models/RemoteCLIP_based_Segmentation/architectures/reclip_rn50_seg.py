import torch.nn as nn  
import torch.nn.functional as F  
from ..core.base import BaseRemoteCLIPSeg  

class ReCLIPResNetSeg(BaseRemoteCLIPSeg):  
    def __init__(
        self, 
        model_name, 
        num_classes=9, 
        input_size=224, 
        ckpt_path=None, 
        freeze_clip=True,
        in_channels=4,  
        device_ids=None,
    ):  
        super().__init__(
            model_name, 
            in_channels,
            input_size,
            ckpt_path, 
            freeze_clip,
            device_ids,  
        )  
              
        # 适配conv1输出通道与adapter输入通道  
        conv1_out_c = self.encoder.conv1.out_channels  
        out_c = self.encoder.layer4[-1].conv3.out_channels  
        self.channel_adapter = nn.Sequential(  
            nn.Conv2d(conv1_out_c, 64, 1, bias=False),  
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True)  
        ).to(self.main_device)  
        
        self.final_conv = nn.Conv2d(out_c, num_classes, 1).to(self.main_device)  

    def forward(self, x):  
        self._validate_input(x)  
        x = self.encoder.conv1(x)  
        x = self.encoder.bn1(x)  
        x = self.encoder.act1(x)  
        x = self.channel_adapter(x)  
        x = self.encoder.layer1(x)  
        x = self.encoder.layer2(x)  
        x = self.encoder.layer3(x)  
        x = self.encoder.layer4(x)  
        x = self.final_conv(x)  
        if x.shape[-2:] != (self.input_size, self.input_size):  
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)  
        return x  