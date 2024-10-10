import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class UNet(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(UNet, self).__init__()  

        # 定义U-Net的编码器部分  
        self.encoder1 = self.conv_block(in_channels, 64)  
        self.encoder2 = self.conv_block(64, 128)  
        self.encoder3 = self.conv_block(128, 256)  
        self.encoder4 = self.conv_block(256, 512)  

        # 定义U-Net的中心部分  
        self.center = self.conv_block(512, 1024)  

        # 定义U-Net的解码器部分  
        self.decoder4 = self.upconv_block(1024, 512, 512)  
        self.decoder3 = self.upconv_block(512, 256, 256)  
        self.decoder2 = self.upconv_block(256, 128, 128)  
        self.decoder1 = self.upconv_block(128, 64, 64)  

        # 最后的1x1卷积层，用于输出  
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  

        # 初始化权重  
        self.initialize_weights()  

    def conv_block(self, in_channels, out_channels):  
        # 卷积块：两个卷积层 + 批量归一化 + ReLU  
        return nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )  

    def upconv_block(self, in_channels, mid_channels, out_channels):  
        # 上采样然后卷积  
        return nn.Sequential(  
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2),  
            self.conv_block(mid_channels + mid_channels, out_channels)  # 合并后的结果送入卷积块  
        )  

    def forward(self, x):  
        # 编码器  
        enc1 = self.encoder1(x)  # 输出: [Batch, 64, H, W]  
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))  # 输出: [Batch, 128, H/2, W/2]  
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))  # 输出: [Batch, 256, H/4, W/4]  
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))  # 输出: [Batch, 512, H/8, W/8]  

        # 中心部分  
        center = self.center(F.max_pool2d(enc4, kernel_size=2))  # 输出: [Batch, 1024, H/16, W/16]  

        # 解码器  
        dec4 = self.decoder4(center)  # 上采样中心，初始期望输出为 [Batch, 512, H/8, W/8]  
        dec4 = torch.cat((dec4, enc4), dim=1)  # 与enc4连接: [Batch, 1024, H/8, W/8]  

        dec3 = self.decoder3(dec4)  # 期望输入为1024通道，将输出 [Batch, 256, H/4, W/4]  
        dec3 = torch.cat((dec3, enc3), dim=1)  # 与enc3连接: [Batch, 512, H/4, W/4]  

        dec2 = self.decoder2(dec3)  # 期望输入为512通道，将输出 [Batch, 128, H/2, W/2]  
        dec2 = torch.cat((dec2, enc2), dim=1)  # 与enc2连接: [Batch, 256, H/2, W/2]  

        dec1 = self.decoder1(dec2)  # 期望输入为256通道，将输出 [Batch, 64, H, W]  
        dec1 = torch.cat((dec1, enc1), dim=1)  # 与enc1连接: [Batch, 128, H, W]  

        # 输出层  
        return self.final_conv(dec1)  # 输出到所需通道  

    def initialize_weights(self):  
        # 使用He初始化  
        for m in self.modules():  
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):  
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                if m.bias is not None:  
                    nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)