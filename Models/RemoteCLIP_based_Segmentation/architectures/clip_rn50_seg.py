import torch.nn as nn  
import torch.nn.functional as F  
from ..core.base import BaseCLIPSegmentation  

class CLIPResNetSegmentation(BaseCLIPSegmentation):  
    def __init__(self, model_name, num_classes=9, input_size=224, ckpt_path=None, freeze_clip=True):  
        super().__init__(model_name, num_classes, input_size, ckpt_path, freeze_clip)  
        out_c = self.visual_encoder.layer4[-1].conv3.out_channels  
        self.channel_adapter = nn.Sequential(  
            nn.Conv2d(32, 64, 1, bias=False),  
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True)  
        ).to(self.device)  
        self.final_conv = nn.Conv2d(out_c, num_classes, 1).to(self.device)  

    def forward(self, x):  
        self._validate_input(x)  
        x = x.to(self.device)  
        x = self.visual_encoder.conv1(x)  
        x = self.visual_encoder.bn1(x)  
        x = self.visual_encoder.act1(x)  
        x = self.channel_adapter(x)  
        x = self.visual_encoder.layer1(x)  
        x = self.visual_encoder.layer2(x)  
        x = self.visual_encoder.layer3(x)  
        x = self.visual_encoder.layer4(x)  
        x = self.final_conv(x)  
        if x.shape[-2:] != (self.input_size, self.input_size):  
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)  
        return x  