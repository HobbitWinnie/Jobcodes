import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DoubleConv(nn.Module):
    """双重卷积块"""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionGate(nn.Module):
    """注意力门控模块"""
    def __init__(self, F_g: int, F_l: int, F_int: Optional[int] = None):
        super().__init__()
        if F_int is None:
            F_int = min(F_g, F_l)
        else:
            F_int = min(F_int, min(F_g, F_l))

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)

class Down(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2, bilinear: bool = False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
        
        self.attention = AttentionGate(F_g=in_channels//2, F_l=in_channels//2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理输入大小差异
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        # 应用注意力机制
        x2_att = self.attention(g=x1, x=x2)
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, 
                 in_channels: int = 4, 
                 out_channels: int = 8, 
                 initial_features: int = 64,
                 dropout_rate: float = 0.2,
                 bilinear: bool = False):
        """
        增强型UNet网络
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            initial_features: 初始特征数
            dropout_rate: dropout比率
            bilinear: 是否使用双线性插值上采样
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        
        # Encoder path
        self.inc = DoubleConv(in_channels, initial_features, dropout_rate)
        self.down1 = Down(initial_features, initial_features*2, dropout_rate)
        self.down2 = Down(initial_features*2, initial_features*4, dropout_rate)
        self.down3 = Down(initial_features*4, initial_features*8, dropout_rate)
        self.down4 = Down(initial_features*8, initial_features*16//factor, dropout_rate)
        
        # Decoder path
        self.up1 = Up(initial_features*16, initial_features*8//factor, dropout_rate, bilinear)
        self.up2 = Up(initial_features*8, initial_features*4//factor, dropout_rate, bilinear)
        self.up3 = Up(initial_features*4, initial_features*2//factor, dropout_rate, bilinear)
        self.up4 = Up(initial_features*2, initial_features, dropout_rate, bilinear)
        
        # Output layer
        self.outc = nn.Sequential(
            nn.Conv2d(initial_features, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.initialize_weights()

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)

    def initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)