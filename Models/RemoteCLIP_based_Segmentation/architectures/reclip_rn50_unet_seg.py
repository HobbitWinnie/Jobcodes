import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from ..core.base import BaseRemoteCLIPSeg 

class DoubleConv(nn.Module):  
    """双卷积块"""  
    def __init__(self, in_channels, out_channels, mid_channels=None):  
        super().__init__()  
        if not mid_channels:  
            mid_channels = out_channels  
        self.double_conv = nn.Sequential(  
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(mid_channels),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),  
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
        if skip is not None:  
            # 与主干尺寸对齐  
            if x.shape[-2:] != skip.shape[-2:]:  
                skip = F.interpolate(  
                    skip,  
                    size=x.shape[-2:],  
                    mode='bilinear',  
                    align_corners=False  
                )  
            x = torch.cat([skip, x], dim=1)  
        x = self.dropout(x)  
        return self.conv(x)  

class UNetWithReCLIPResNet(BaseRemoteCLIPSeg):  
    def __init__(
        self, 
        model_name, 
        ckpt_path, 
        num_classes, 
        input_size=224,  
        dropout_rate=0.1, 
        use_aux_loss=True, 
        initial_features=64,
        in_channels=3,  
        device_ids=None,  
    ):  
        super().__init__(
            model_name, 
            num_classes, 
            input_size, 
            ckpt_path, 
            freeze_clip=True, 
            in_channels=in_channels,  
            device_ids=device_ids,  
        )  
        self.use_aux_loss = use_aux_loss  
        self.input_size = input_size  
        self.main_device = self.main_device  # 统一命名  

        # 获取每一层输出通道配置  
        encoder_layers = self._get_visual_encoder_layers()  
        # 用一组1x1卷积将主干输出统一至U-Net的维度  
        self.feature_transforms = nn.ModuleList([  
            nn.Conv2d(layer[1], initial_features * (2 ** i), kernel_size=1)  
            for i, layer in enumerate(encoder_layers)  
        ])  
        decoder_in_channels = [initial_features * (2 ** i) for i in range(4, 0, -1)]  
        decoder_skip_channels = [initial_features * (2 ** i) for i in range(3, -1, -1)]  
        decoder_out_channels = [initial_features * (2 ** i) for i in range(3, -1, -1)]  
        self.decoder_blocks = nn.ModuleList([  
            DecoderBlock(in_ch, skip_ch, out_ch, dropout_rate)  
            for in_ch, skip_ch, out_ch in zip(  
                decoder_in_channels, decoder_skip_channels, decoder_out_channels  
            )  
        ])  
        self.final_conv = nn.Sequential(  
            nn.Conv2d(initial_features, initial_features // 2, 3, padding=1, bias=False),  
            nn.BatchNorm2d(initial_features // 2),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            nn.Conv2d(initial_features // 2, num_classes, 1)  
        )  
        if use_aux_loss:  
            self.aux_head = nn.Sequential(  
                nn.Conv2d(initial_features * 16, initial_features * 8, 3, padding=1, bias=False),  
                nn.BatchNorm2d(initial_features * 8),  
                nn.ReLU(inplace=True),  
                nn.Dropout2d(dropout_rate),  
                nn.Conv2d(initial_features * 8, num_classes, 1)  
            )  
        self.initialize_weights()  

    def _get_visual_encoder_layers(self):  
        # 获取视觉编码器每层名称与输出通道数  
        # 仅适配clip resnet主干  
        ve = self.encoder    
        return [  
            ('conv1', ve.conv1.out_channels),  
            ('layer1', ve.layer1[-1].conv3.out_channels),  
            ('layer2', ve.layer2[-1].conv3.out_channels),  
            ('layer3', ve.layer3[-1].conv3.out_channels),  
            ('layer4', ve.layer4[-1].conv3.out_channels)  
        ]  

    def extract_features(self, x):  
        # 按照resnet主干结构提特征  
        features = []  
        x = x.to(self.main_device)  

        with torch.no_grad():  
            self.encoder.eval()  
            x = self.encoder.conv1(x)  # 224 -> 112  
            x = self.encoder.bn1(x)  
            x = self.encoder.act1(x)  
            features.append(x)  
            x = self.encoder.layer1(x)  # 112 -> 112  
            features.append(x)  
            x = self.encoder.layer2(x)  # 112 -> 56  
            features.append(x)  
            x = self.encoder.layer3(x)  # 56 -> 28  
            features.append(x)  
            x = self.encoder.layer4(x)  # 28 -> 14  
            features.append(x)  
        return features  

    def forward(self, x):  
        self._validate_input(x)  
        features = self.extract_features(x)  
        features = [  
            transform(f) for transform, f in zip(self.feature_transforms, features)  
        ]  
        skips = features[:-1]  
        x = features[-1]  
        aux_feature = x.clone() if self.use_aux_loss else None  
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):  
            x = decoder_block(x, skip)  
        main_output = self.final_conv(x)  
        if main_output.shape[-2:] != (self.input_size, self.input_size):  
            main_output = F.interpolate(  
                main_output, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False  
            )  
        outputs = {'main': main_output}  
        if self.use_aux_loss and aux_feature is not None:  
            aux_out = self.aux_head(aux_feature)  
            aux_out = F.interpolate(  
                aux_out, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False  
            )  
            outputs['aux'] = aux_out  
        return outputs  

    def train(self, mode=True):  
        super().train(mode)  
        if hasattr(self, 'visual_encoder'):  
            self.visual_encoder.eval()  
        return self  

    def initialize_weights(self):  
        visual_encoder_modules = set(list(self.visual_encoder.modules()))  
        for m in self.modules():  
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):  
                if m not in visual_encoder_modules:  
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                    if m.bias is not None:  
                        nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):  
                if m not in visual_encoder_modules:  
                    nn.init.constant_(m.weight, 1)  
                    nn.init.constant_(m.bias, 0)  