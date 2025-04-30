import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from ..core.base import BaseRemoteCLIPSeg  

class DoubleConv(nn.Module):  
    """双卷积块"""
    def __init__(self, in_channels, out_channels, mid_channels=None):  
        super().__init__()  
        mid_channels = mid_channels or out_channels  
        self.double_conv = nn.Sequential(  
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),  
            nn.BatchNorm2d(mid_channels),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )  

    def forward(self, x):  
        return self.double_conv(x)  

class DecoderBlock(nn.Module):
    """解码器块"""
    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x, skip=None):
        x = self.up(x)
        # 确保skip和x的空间尺寸匹配
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
        x = self.dropout(x)
        return self.conv(x)
    
class UNetWithReCLIPResNet(BaseRemoteCLIPSeg):  
    """典型UNet-ResNet主干+CLIP特征头"""  
    def __init__(  
        self, 
        model_name, 
        num_classes, 
        input_size=224,  
        in_channels=4, 
        dropout_rate=0.1, 
        ckpt_path=None,  
        freeze_clip=True, 
        use_aux_loss=True, 
        device_ids=None, 
        logger=None  
    ):  
        super().__init__(
            model_name, 
            in_channels, 
            input_size, 
            ckpt_path, 
            freeze_clip, 
            device_ids, 
            logger
        )  
        self.use_aux_loss = use_aux_loss  

        # 1. 动态获取编码器层信息
        encoder_channels = self.encoder_channels   

        # 2. 特征变换层（不变宽）
        self.feature_transforms = nn.ModuleList([  
            nn.Conv2d(c, c, 1) for c in encoder_channels  
        ])  

        # 3. 解码器
        decoder_in = encoder_channels[::-1][:-1]  
        decoder_skip = encoder_channels[::-1][1:]  
        decoder_out = decoder_skip  
        self.decoder_blocks = nn.ModuleList([  
            DecoderBlock(in_ch, skip_ch, out_ch, dropout_rate)  
            for in_ch, skip_ch, out_ch in zip(
                decoder_in, decoder_skip, decoder_out
            )  
        ])  

        # 4. 最终卷积
        self.final_conv = nn.Sequential(  
            nn.Conv2d(decoder_out[-1], decoder_out[-1] // 2, 3, padding=1, bias=False),  
            nn.BatchNorm2d(decoder_out[-1] // 2),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            nn.Conv2d(decoder_out[-1] // 2, num_classes, 1)  
        )  

        # 5. 辅助头（用encoder最后一层的通道）
        if use_aux_loss:  
            self.aux_head = nn.Sequential(  
                nn.Conv2d(encoder_channels[-1], encoder_channels[-1] // 2, 3, padding=1, bias=False),  
                nn.BatchNorm2d(encoder_channels[-1] // 2),  
                nn.ReLU(inplace=True),  
                nn.Dropout2d(dropout_rate),  
                nn.Conv2d(encoder_channels[-1] // 2, num_classes, 1)  
            )  
        self.initialize_weights()  
        super().to(self.main_device)  

    def forward(self, x):  
        # 数据验证
        self._validate_input(x)  

        # with torch.cuda.amp.autocast():  
            # 提取各层特征
        features = self.extract_encoder_features(x)  
            # 特征变换
            # for t, f in zip(self.feature_transforms, features):  
            #     print("feature:", f.dtype, "weight:", t.weight.dtype, "bias:", t.bias.dtype if t.bias is not None else None)  
            
        features = [t(f) for t, f in zip(self.feature_transforms, features)]  

        # 跳跃连接和decode
        skips = features[:-1]  
        x = features[-1]  
        aux_feature = x.clone() if self.use_aux_loss else None  
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):  
            x = decoder_block(x, skip)  
        main_output = self.final_conv(x)  
        if main_output.shape[-2:] != (self.input_size, self.input_size):  
            main_output = F.interpolate(  
                main_output,  
                size=(self.input_size, self.input_size),  
                mode='bilinear',  
                align_corners=False  
            )  
        outputs = {'main': main_output}  
        if self.use_aux_loss and aux_feature is not None:  
            aux_output = self.aux_head(aux_feature)  
            outputs['aux'] = F.interpolate(  
                aux_output,  
                size=(self.input_size, self.input_size),  
                mode='bilinear',  
                align_corners=False  
            )  
        return outputs  

    def initialize_weights(self):  
        encoder_modules = list(self.encoder.modules())  
        for m in self.modules():  
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):  
                if m not in encoder_modules:  
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                    if m.bias is not None:  
                        nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):  
                if m not in encoder_modules:  
                    nn.init.constant_(m.weight, 1)  
                    nn.init.constant_(m.bias, 0) 