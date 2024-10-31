import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from typing import Optional, Dict, Union, Tuple  
from transformers import CLIPVisionModel  

class DoubleConv(nn.Module):  
    """双卷积模块"""  
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):  
        super().__init__()  
        self.double_conv = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(p=dropout_rate),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(p=dropout_rate)  
        )  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.double_conv(x)  

class AttentionGate(nn.Module):  
    """改进的注意力门控模块"""  
    def __init__(self, F_g: int, F_l: int, F_int: int):  
        super().__init__()  
        self.W_g = nn.Sequential(  
            nn.Conv2d(F_g, F_int, kernel_size=1),  
            nn.BatchNorm2d(F_int)  
        )  
        self.W_x = nn.Sequential(  
            nn.Conv2d(F_l, F_int, kernel_size=1),  
            nn.BatchNorm2d(F_int)  
        )  
        self.psi = nn.Sequential(  
            nn.Conv2d(F_int, 1, kernel_size=1),  
            nn.BatchNorm2d(1),  
            nn.Sigmoid()  
        )  
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  
        g1 = self.W_g(g)  
        x1 = self.W_x(x)  
        psi = self.relu(g1 + x1)  
        psi = self.psi(psi)  
        return x * psi  

class SpatialAttention(nn.Module):  
    """空间注意力模块"""  
    def __init__(self, kernel_size: int = 7):  
        super().__init__()  
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        x = torch.cat([avg_out, max_out], dim=1)  
        x = self.conv(x)  
        return self.sigmoid(x)  

class ChannelAttention(nn.Module):  
    """通道注意力模块"""  
    def __init__(self, in_channels: int, ratio: int = 16):  
        super().__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1)  
        self.relu = nn.ReLU(inplace=True)  
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))  
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))  
        out = avg_out + max_out  
        return self.sigmoid(out)  

class Down(nn.Module):  
    """下采样模块"""  
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):  
        super().__init__()  
        self.maxpool_conv = nn.Sequential(  
            nn.MaxPool2d(2),  
            DoubleConv(in_channels, out_channels, dropout_rate)  
        )  
        self.channel_attention = ChannelAttention(out_channels)  
        self.spatial_attention = SpatialAttention()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x = self.maxpool_conv(x)  
        ca = self.channel_attention(x)  
        sa = self.spatial_attention(x)  
        return x * ca * sa  

class Up(nn.Module):  
    """上采样模块"""  
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2,   
                 bilinear: bool = False):  
        super().__init__()  
        if bilinear:  
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)  
        else:  
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)  
        
        self.attention = AttentionGate(F_g=in_channels//2, F_l=in_channels//2, F_int=in_channels//4)  

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  
        x1 = self.up(x1)  
        
        # 处理输入尺寸不匹配  
        diff_y = x2.size()[2] - x1.size()[2]  
        diff_x = x2.size()[3] - x1.size()[3]  
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,  
                       diff_y // 2, diff_y - diff_y // 2])  
        
        # 应用注意力机制  
        x2_attended = self.attention(g=x1, x=x2)  
        x = torch.cat([x2_attended, x1], dim=1)  
        return self.conv(x)  

class RomoClipUNet(nn.Module):  
    def __init__(self,   
                 in_channels: int = 4,   
                 out_channels: int = 8,  
                 initial_features: int = 64,  
                 dropout_rate: float = 0.2,  
                 bilinear: bool = False,  
                 clip_model_name: str = "MVRL/RomoClip",  
                 freeze_clip: bool = True):  
        super().__init__()  
        self.in_channels = in_channels  
        self.out_channels = out_channels  
        self.bilinear = bilinear  
        factor = 2 if bilinear else 1  

        # CLIP视觉编码器  
        self.clip_encoder = CLIPVisionModel.from_pretrained(clip_model_name)  
        if freeze_clip:  
            for param in self.clip_encoder.parameters():  
                param.requires_grad = False  

        # CLIP特征转换  
        self.clip_proj = nn.Sequential(  
            nn.Conv2d(768, initial_features*16//factor, kernel_size=1),  
            nn.BatchNorm2d(initial_features*16//factor),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate)  
        )  

        # 输入适配  
        self.input_adapt = None  
        if in_channels != 3:  
            self.input_adapt = nn.Conv2d(in_channels, 3, kernel_size=1)  

        # Encoder  
        self.inc = DoubleConv(in_channels, initial_features, dropout_rate)  
        self.down1 = Down(initial_features, initial_features*2, dropout_rate)  
        self.down2 = Down(initial_features*2, initial_features*4, dropout_rate)  
        self.down3 = Down(initial_features*4, initial_features*8, dropout_rate)  
        self.down4 = Down(initial_features*8, initial_features*16//factor, dropout_rate)  

        # 特征融合  
        self.fusion = nn.Sequential(  
            nn.Conv2d(initial_features*32//factor, initial_features*16//factor, 1),  
            nn.BatchNorm2d(initial_features*16//factor),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate)  
        )  

        # Decoder  
        self.up1 = Up(initial_features*16, initial_features*8//factor, dropout_rate, bilinear)  
        self.up2 = Up(initial_features*8, initial_features*4//factor, dropout_rate, bilinear)  
        self.up3 = Up(initial_features*4, initial_features*2//factor, dropout_rate, bilinear)  
        self.up4 = Up(initial_features*2, initial_features, dropout_rate, bilinear)  

        # 输出层  
        self.outc = nn.Sequential(  
            nn.Conv2d(initial_features, out_channels, kernel_size=1),  
            nn.BatchNorm2d(out_channels)  
        )  

        # 辅助分类头  
        self.aux_head = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),  
            nn.Linear(initial_features*16//factor, out_channels)  
        )  

        self.initialize_weights()  

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  
        # CLIP特征提取  
        if self.input_adapt is not None:  
            clip_input = self.input_adapt(x)  
        else:  
            clip_input = x  

        clip_input = F.interpolate(clip_input, size=(224, 224),   
                                 mode='bilinear', align_corners=False)  
        
        clip_features = self.clip_encoder(clip_input).last_hidden_state  
        batch_size = clip_features.shape[0]  
        clip_features = clip_features.reshape(batch_size, 768, 14, 14)  
        clip_features = self.clip_proj(clip_features)  

        # Encoder路径  
        x1 = self.inc(x)  
        x2 = self.down1(x1)  
        x3 = self.down2(x2)  
        x4 = self.down3(x3)  
        x5 = self.down4(x4)  

        # 特征融合  
        clip_features = F.interpolate(clip_features, size=x5.shape[2:],  
                                    mode='bilinear', align_corners=False)  
        fused_features = self.fusion(torch.cat([clip_features, x5], dim=1))  

        # Decoder路径  
        x = self.up1(fused_features, x4)  
        x = self.up2(x, x3)  
        x = self.up3(x, x2)  
        x = self.up4(x, x1)  
        
        # 主要输出  
        logits = self.outc(x)  
        
        # 辅助输出  
        aux_out = self.aux_head(fused_features)  

        return {  
            'segmentation': logits,  
            'auxiliary': aux_out,  
            'features': fused_features,  
            'clip_features': clip_features  
        }  

    def initialize_weights(self):  
        """初始化新增加的层的权重"""  
        for m in self.modules():  
            if isinstance(m, (nn.Conv2d, nn.Linear)):  
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                if m.bias is not None:  
                    nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)