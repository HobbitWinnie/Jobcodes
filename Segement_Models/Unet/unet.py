import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class UNet(nn.Module):  
    def __init__(self, in_channels=4, out_channels=8):  # 设定类别数量  
        super(UNet, self).__init__()  

        # Encoder  
        self.encoder1 = self.conv_block(in_channels, 64)  
        self.encoder2 = self.conv_block(64, 128)  
        self.encoder3 = self.conv_block(128, 256)  
        self.encoder4 = self.conv_block(256, 512)  

        # Center  
        self.center = self.conv_block(512, 1024)  

        # Decoder  
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  
        self.decoder4 = self.conv_block(1024, 512)  # 使用 1024 作为输入通道数  

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  
        self.decoder3 = self.conv_block(512, 256)  

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  
        self.decoder2 = self.conv_block(256, 128)  

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  
        self.decoder1 = self.conv_block(128, 64)  

        # Final 1x1 conv layer for output  
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  

        # Initialize weights  
        self._initialize_weights()  


    def conv_block(self, in_channels, out_channels):  
        return nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )  

    def forward(self, x):  
        # Encoder  
        enc1 = self.encoder1(x)  
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))  
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))  
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))  

        # Center  
        center = self.center(F.max_pool2d(enc4, 2))  

        # Decoder  
        up4 = self.upconv4(center)  
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))  

        up3 = self.upconv3(dec4)  
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))  

        up2 = self.upconv2(dec3)  
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))  

        up1 = self.upconv1(dec2)  
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))  

        return self.final_conv(dec1)  
    
    def _initialize_weights(self):  
        # 使用He (Kaiming) 初始化  
        for m in self.modules():  
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):  
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)  

# # Testing the model with a sample input  
# model = UNet(in_channels=4, out_channels=8)  # 假设有8个类别  
# input_tensor = torch.randn(192, 4, 256, 256)  # 批次大小192  
# output = model(input_tensor)  
# print(output.shape)  # 应该是 [192, 8, 256, 256]