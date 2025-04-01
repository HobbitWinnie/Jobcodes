import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# 定义全局卷积长短期记忆网络（GCL）模块（时间步长设为8，卷积核大小设为5）  
class GCL(nn.Module):  
    def __init__(self, input_channels, hidden_channels, kernel_size=5, time_steps=8):  
        super(GCL, self).__init__()  
        self.time_steps = time_steps  
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2)  
        self.bn = nn.BatchNorm2d(hidden_channels)  
        self.relu = nn.ReLU(inplace=True)  
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(hidden_channels, hidden_channels) for _ in range(2)])  

    def forward(self, x):  
        batch_size, c, h, w = x.size()  
        x = self.relu(self.bn(self.conv(x)))  # 经过卷积层  
        x = x.view(batch_size, c, -1).permute(2, 0, 1)  # 转换为序列 [seq_len, batch_size, feature_dim]  
        h_t = [torch.zeros(batch_size, c, device=x.device) for _ in range(2)]  
        c_t = [torch.zeros(batch_size, c, device=x.device) for _ in range(2)]  
        outputs = []  
        for time in range(self.time_steps):  
            for i, lstm_cell in enumerate(self.lstm_cells):  
                if i == 0:  
                    h_t[i], c_t[i] = lstm_cell(x[time % x.size(0)], (h_t[i], c_t[i]))  
                else:  
                    h_t[i], c_t[i] = lstm_cell(h_t[i - 1], (h_t[i], c_t[i]))  
            outputs.append(h_t[-1])  
        x = torch.stack(outputs, dim=0).permute(1, 2, 0).view(batch_size, c, h, w)  
        return x  

# 定义全局联合注意力机制（GJAM）模块  
class GJAM(nn.Module):  
    def __init__(self, in_channels):  
        super(GJAM, self).__init__()  
        # 光谱（通道）注意力机制  
        self.channel_attention = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(in_channels, in_channels // 8, 1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_channels // 8, in_channels, 1),  
            nn.Sigmoid()  
        )  
        # 空间注意力机制  
        self.spatial_attention = nn.Sequential(  
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  
            nn.Sigmoid()  
        )  

    def forward(self, x):  
        # 光谱注意力  
        ca = self.channel_attention(x)  
        x = x * ca  

        # 空间注意力  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        sa = torch.cat([max_out, avg_out], dim=1)  
        sa = self.spatial_attention(sa)  
        x = x * sa  

        return x  

# 定义SSDGL模型（基于编码器-解码器架构）  
class SSDGLNet(nn.Module):  
    def __init__(self, num_classes):  
        super(SSDGLNet, self).__init__()  
        # 编码器  
        self.encoder = nn.Sequential(  
            nn.Conv2d(1, 64, 3, padding=1),  # 假设输入通道为1  
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(2, 2),  # 下采样  

            nn.Conv2d(64, 128, 3, padding=1),  
            nn.BatchNorm2d(128),  
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(128, 256, 3, padding=1),  
            nn.BatchNorm2d(256),  
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(2, 2)  
        )  
        # GCL模块  
        self.gcl = GCL(input_channels=256, hidden_channels=256, kernel_size=5, time_steps=8)  
        # GJAM模块  
        self.gjam = GJAM(in_channels=256)  
        # 解码器  
        self.decoder = nn.Sequential(  
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 上采样  
            nn.BatchNorm2d(128),  
            nn.ReLU(inplace=True),  

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True),  

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True),  

            nn.Conv2d(64, num_classes, kernel_size=1)  
        )  

    def forward(self, x):  
        x = self.encoder(x)  
        x = self.gcl(x)  
        x = self.gjam(x)  
        x = self.decoder(x)  
        return x  