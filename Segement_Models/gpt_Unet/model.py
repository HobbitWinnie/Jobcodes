import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class AttentionBlock(nn.Module):  
    def __init__(self, F_g, F_l, F_int):  
        super(AttentionBlock, self).__init__()  
        #确保中间通道数不会太大  
        F_int = F_int if F_int <= min(F_g, F_l) else min(F_g, F_l)  
        
        self.W_g = nn.Sequential(  
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  
            nn.BatchNorm2d(F_int)  
        )  
        
        self.W_x = nn.Sequential(  
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),  
            nn.BatchNorm2d(F_int)  
        )  
        
        self.psi = nn.Sequential(  
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),  
            nn.BatchNorm2d(1),  
            nn.Sigmoid()  
        )  
        
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, g, x):  
        g1 = self.W_g(g)  
        x1 = self.W_x(x)  
        psi = self.relu(g1 + x1)  
        psi = self.psi(psi)  
        return x * psi  

class UNet(nn.Module):  
    def __init__(self, in_channels=4, out_channels=8, dropout_rate=0.2, initial_features=64):  
        super(UNet, self).__init__()  
        self.initial_features = initial_features  
        features = initial_features  
        
        # Encoder  
        self.encoder1 = self.conv_block(in_channels, features, dropout_rate)  
        self.encoder2 = self.conv_block(features, features*2, dropout_rate)  
        self.encoder3 = self.conv_block(features*2, features*4, dropout_rate)  
        self.encoder4 = self.conv_block(features*4, features*8, dropout_rate)  

        # Center  
        self.center = self.conv_block(features*8, features*16, dropout_rate)  

        # Attention blocks - 修正通道数  
        self.attention4 = AttentionBlock(F_g=features*8, F_l=features*8, F_int=features*4)  
        self.attention3 = AttentionBlock(F_g=features*4, F_l=features*4, F_int=features*2)  
        self.attention2 = AttentionBlock(F_g=features*2, F_l=features*2, F_int=features)  
        self.attention1 = AttentionBlock(F_g=features, F_l=features, F_int=features//2)  

        # Decoder  
        self.upconv4 = nn.Sequential(  
            nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2),  
            nn.BatchNorm2d(features*8),  
            nn.ReLU(inplace=True)  
        )  
        self.decoder4 = self.conv_block(features*16, features*8, dropout_rate)  

        self.upconv3 = nn.Sequential(  
            nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2),  
            nn.BatchNorm2d(features*4),  
            nn.ReLU(inplace=True)  
        )  
        self.decoder3 = self.conv_block(features*8, features*4, dropout_rate)  

        self.upconv2 = nn.Sequential(  
            nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2),  
            nn.BatchNorm2d(features*2),  
            nn.ReLU(inplace=True)  
        )  
        self.decoder2 = self.conv_block(features*4, features*2, dropout_rate)  

        self.upconv1 = nn.Sequential(  
            nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2),  
            nn.BatchNorm2d(features),  
            nn.ReLU(inplace=True)  
        )  
        self.decoder1 = self.conv_block(features*2, features, dropout_rate)  

        # Final layers  
        self.final_conv = nn.Sequential(  
            nn.Conv2d(features, features, kernel_size=3, padding=1),  
            nn.BatchNorm2d(features),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            nn.Conv2d(features, out_channels, kernel_size=1)  
        )  

        self._initialize_weights()  

    def conv_block(self, in_channels, out_channels, dropout_rate):  
        return nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate)  
        )  

    def forward(self, x):  
        # Encoder path  
        enc1 = self.encoder1(x)  
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))  
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))  
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))  

        # Center  
        center = self.center(F.max_pool2d(enc4, 2))  

        # Decoder path with attention  
        dec4 = self.upconv4(center)  
        enc4_att = self.attention4(g=dec4, x=enc4)  
        dec4 = self.decoder4(torch.cat([dec4, enc4_att], dim=1))  

        dec3 = self.upconv3(dec4)  
        enc3_att = self.attention3(g=dec3, x=enc3)  
        dec3 = self.decoder3(torch.cat([dec3, enc3_att], dim=1))  

        dec2 = self.upconv2(dec3)  
        enc2_att = self.attention2(g=dec2, x=enc2)  
        dec2 = self.decoder2(torch.cat([dec2, enc2_att], dim=1))  

        dec1 = self.upconv1(dec2)  
        enc1_att = self.attention1(g=dec1, x=enc1)  
        dec1 = self.decoder1(torch.cat([dec1, enc1_att], dim=1))  

        return self.final_conv(dec1)  

    def _initialize_weights(self):  
        for m in self.modules():  
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):  
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                if m.bias is not None:  
                    nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)