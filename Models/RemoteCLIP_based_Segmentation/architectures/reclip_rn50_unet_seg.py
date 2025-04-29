import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from ..core.base import BaseRemoteCLIPSeg   

class DoubleConv(nn.Module):  
    """双卷积块"""  
    def __init__(self, in_channels, out_channels, mid_channels=None):  
        super().__init__()  
        mid_channels = out_channels if mid_channels is None else mid_channels  
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
    """解码器块（通道数适配ResNet50）"""  
    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.1):  
        super().__init__()  
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)  
        self.dropout = nn.Dropout2d(dropout_rate)  
    
    def forward(self, x, skip=None):  
        x = self.up(x)  
        if skip is not None:  
            if x.shape[-2:] != skip.shape[-2:]:  
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)  
            x = torch.cat([skip, x], dim=1)  
        x = self.dropout(x)  
        return self.conv(x)  

class UNetWithReCLIPResNet(BaseRemoteCLIPSeg):  
    def __init__(  
        self,   
        model_name,   
        num_classes=9,   
        input_size=224,   
        ckpt_path=None,   
        freeze_clip=True,  
        in_channels=4,  
        device_ids=None,  
        dropout_rate=0.1,   
        use_aux_loss=True,   
        initial_features=64,  # 必须与conv1.out_channels一致  
    ):  
        super().__init__(  
            model_name=model_name,  
            num_classes=num_classes,  
            input_size=input_size,  
            ckpt_path=ckpt_path,  
            freeze_clip=freeze_clip,  
            in_channels=in_channels,  
            device_ids=device_ids,  
        )  
        self.use_aux_loss = use_aux_loss  
        self.input_size = input_size  

        # 获取主干实际通道配置  
        encoder_layers = self._get_visual_encoder_layers()  
        print(f"[DEBUG] Encoder layers: {encoder_layers}")  # 验证通道数  

        assert encoder_layers[0][1] == initial_features, \
            f"主干预期输出通道{initial_features}, 实际{encoder_layers[0][1]}"  

        # 特征转换层（关键修改点）  
        self.feature_transforms = nn.ModuleList([  
            nn.Conv2d(  
                layer_ch,  # 输入通道来自主干实际输出  
                initial_features * (2 ** i),  # i从0开始  
                kernel_size=1  
            )  
            for i, (_, layer_ch) in enumerate(encoder_layers)  
        ])  

        # 解码器通道参数（与特征转换输出对齐）  
        decoder_in_channels = [initial_features * (2 ** i) for i in range(4, 0, -1)]  
        decoder_skip_channels = [initial_features * (2 ** i) for i in range(3, -1, -1)]  
        decoder_out_channels = [initial_features * (2 ** i) for i in range(3, -1, -1)]  
       
        self.decoder_blocks = nn.ModuleList([  
            DecoderBlock(in_ch, skip_ch, out_ch, dropout_rate)  
            for in_ch, skip_ch, out_ch in zip(  
                decoder_in_channels, decoder_skip_channels, decoder_out_channels  
            )  
        ])  

        # 最终输出层  
        self.final_conv = nn.Sequential(  
            nn.Conv2d(256, 128, 3, padding=1, bias=False),  
            nn.BatchNorm2d(128),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            nn.Conv2d(128, num_classes, 1)  
        )  

        # 辅助头（适配layer4的2048→1024）  
        if use_aux_loss:  
            self.aux_head = nn.Sequential(  
                nn.Conv2d(8192, 4096, 3, padding=1, bias=False),  
                nn.BatchNorm2d(4096),  
                nn.ReLU(inplace=True),  
                nn.Dropout2d(dropout_rate),  
                nn.Conv2d(4096, num_classes, 1)  
            )   

        self.initialize_weights()  

    def _get_visual_encoder_layers(self):  
        ve = self.encoder    
        return [  
            ('conv1', ve.conv1.out_channels),  
            ('layer1', ve.layer1[-1].conv3.out_channels),  
            ('layer2', ve.layer2[-1].conv3.out_channels),  
            ('layer3', ve.layer3[-1].conv3.out_channels),  
            ('layer4', ve.layer4[-1].conv3.out_channels)  
        ]  

    def extract_features(self, x, use_grad=False):  
        features = []  
        x = x.to(self.main_device)  
        context = torch.enable_grad() if use_grad else torch.no_grad()  
        with context:  
            # 特征提取流程保持不变  
            x = self.encoder.conv1(x)  
            x = self.encoder.bn1(x)  
            x = self.encoder.act1(x)  
            features.append(x)  
            x = self.encoder.layer1(x)  
            features.append(x)  
            x = self.encoder.layer2(x)  
            features.append(x)  
            x = self.encoder.layer3(x)  
            features.append(x)  
            x = self.encoder.layer4(x)  
            features.append(x)  
        return features  

    def forward(self, x):  
        self._validate_input(x)  
        features = self.extract_features(x, use_grad=self.training and not self.encoder_is_frozen())  
        
        # 特征转换  
        features = [transform(f) for transform, f in zip(self.feature_transforms, features)]  
        print(f"[DEBUG] Transformed features shapes: {[f.shape for f in features]}")  # 验证转换后形状  
        
        skips = features[:-1]  
        x = features[-1]  
        aux_feature = x.clone() if self.use_aux_loss else None  
        
        # 解码过程  
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):  
            x = decoder_block(x, skip)  
        
        # 最终输出  
        main_output = self.final_conv(x)  
        if main_output.shape[-2:] != (self.input_size, self.input_size):  
            main_output = F.interpolate(main_output, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)  
        
        outputs = {'main': main_output}  
        if self.use_aux_loss and aux_feature is not None:  
            aux_out = self.aux_head(aux_feature)  
            aux_out = F.interpolate(aux_out, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)  
            outputs['aux'] = aux_out  
        return outputs  

    def train(self, mode=True):  
        super().train(mode)  
        self.encoder.eval()  # 保持主干一直eval（冻结BN，否则影响表现）  
        return self  

    def encoder_is_frozen(self):  
        # 检查主干参数是否被全部冻结（所有requires_grad为False）  
        return all(not p.requires_grad for p in self.encoder.parameters())  
    
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