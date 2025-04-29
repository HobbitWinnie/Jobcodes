
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

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
        # 确保skip和x的空间尺寸匹配
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
        x = self.dropout(x)
        return self.conv(x)

class UNetWithCLIP(nn.Module):
    def __init__(self, model_name, ckpt_path, num_classes, input_size=224,
                dropout_rate=0.1, use_aux_loss=True):
        super().__init__()
        self.input_size = input_size
        self.use_aux_loss = use_aux_loss

        # 1. 初始化CLIP及保证4通道
        self._init_clip_model(model_name, ckpt_path)

        # 2. 动态获取编码器层信息
        encoder_layers = self._get_visual_encoder_layers()
        encoder_channels = [layer[1] for layer in encoder_layers]      # e.g. [32,256,512,1024,2048]

        # 3. 特征变换层（不变宽）
        self.feature_transforms = nn.ModuleList([
            nn.Conv2d(c, c, 1) for c in encoder_channels
        ])

        # 4. 解码器
        decoder_in_channels   = encoder_channels[::-1][:-1]    # e.g. [2048,1024,512,256]
        decoder_skip_channels = encoder_channels[::-1][1:]     # e.g. [1024,512,256,32]
        decoder_out_channels  = decoder_skip_channels          # e.g. [1024,512,256,32]
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, dropout_rate)
            for in_ch, skip_ch, out_ch in zip(
                decoder_in_channels, decoder_skip_channels, decoder_out_channels
            )
        ])

        # 5. 最终卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_out_channels[-1], decoder_out_channels[-1] // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_out_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(decoder_out_channels[-1] // 2, num_classes, 1)
        )

        # 6. 辅助头（用encoder最后一层的通道）
        if use_aux_loss:
            self.aux_head = nn.Sequential(
                nn.Conv2d(encoder_channels[-1], encoder_channels[-1] // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(encoder_channels[-1] // 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(encoder_channels[-1] // 2, num_classes, 1)
            )
        self.initialize_weights()

    def _init_clip_model(self, model_name, ckpt_path):
        """初始化 CLIP 模型"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, _, _ = open_clip.create_model_and_transforms(model_name)

            if ckpt_path:
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt)

            self.visual_encoder = model.visual
            self.visual_encoder.eval()
            # 确保编码器通道数正确
            self._ensure_encoder_channels()
            # 冻结参数
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"CLIP 模型加载失败: {str(e)}")
            raise RuntimeError(f"CLIP 模型加载失败: {str(e)}")

    def _ensure_encoder_channels(self):  
        original_conv1 = self.visual_encoder.conv1  
        original_bn1 = self.visual_encoder.bn1  
        # 新增：取主干第一个block的输入通道数，确保一致  
        target_out_channels = self.visual_encoder.layer1[0].conv1.in_channels  

        new_conv1 = nn.Conv2d(  
            in_channels=4,  
            out_channels=target_out_channels,  
            kernel_size=original_conv1.kernel_size,  
            stride=original_conv1.stride,  
            padding=original_conv1.padding,  
            bias=False  
        )  
        with torch.no_grad():  
            orig_w = original_conv1.weight.data  # [可能是(64,3,3,3)]  
            new_w = new_conv1.weight.data  
            num_copy = min(orig_w.shape[0], new_w.shape[0])  
            new_w[:num_copy, :3, :, :] = orig_w[:num_copy]  
            new_w[:num_copy, 3, :, :] = orig_w[:num_copy].mean(dim=1)  
        new_conv1.weight.data = new_w  
        self.visual_encoder.conv1 = new_conv1  

        # 同步bn1  
        self.visual_encoder.bn1 = nn.BatchNorm2d(  
            num_features=target_out_channels,  
            eps=original_bn1.eps,  
            momentum=original_bn1.momentum,  
            affine=original_bn1.affine,  
            track_running_stats=original_bn1.track_running_stats  
        )  
        with torch.no_grad():  
            if original_bn1.affine:  
                self.visual_encoder.bn1.weight.data[:num_copy] = original_bn1.weight.data[:num_copy].clone()  
                self.visual_encoder.bn1.bias.data[:num_copy] = original_bn1.bias.data[:num_copy].clone()  
            self.visual_encoder.bn1.running_mean.data[:num_copy] = original_bn1.running_mean.data[:num_copy].clone()  
            self.visual_encoder.bn1.running_var.data[:num_copy] = original_bn1.running_var.data[:num_copy].clone()  


    def _get_visual_encoder_layers(self):
        """获取视觉编码器各层的输出通道数"""
        return [
            ('conv1', self.visual_encoder.conv1.out_channels),
            ('layer1', self.visual_encoder.layer1[-1].conv3.out_channels),
            ('layer2', self.visual_encoder.layer2[-1].conv3.out_channels),
            ('layer3', self.visual_encoder.layer3[-1].conv3.out_channels),
            ('layer4', self.visual_encoder.layer4[-1].conv3.out_channels)
        ]

    def extract_features(self, x):
        """提取CLIP的各层特征"""
        features = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        with torch.no_grad():
            self.visual_encoder.eval()
            # conv1 + bn1 + act1
            x = self.visual_encoder.conv1(x)
            x = self.visual_encoder.bn1(x)
            x = self.visual_encoder.act1(x)
            features.append(x)
            x = self.visual_encoder.layer1(x)
            features.append(x)
            x = self.visual_encoder.layer2(x)
            features.append(x)
            x = self.visual_encoder.layer3(x)
            features.append(x)
            x = self.visual_encoder.layer4(x)
            features.append(x)
        return features

    def forward(self, x):
        self._validate_input(x)
        # 提取各层特征
        features = self.extract_features(x)
        # 特征变换
        features = [
            transform(f)
            for transform, f in zip(self.feature_transforms, features)
        ]
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

    def _validate_input(self, x):
        """验证输入数据"""
        if x.dim() != 4:
            raise ValueError(f"输入应为4维张量，实际维度为{x.dim()}")
        if x.shape[1] != 4:
            raise ValueError(f"期望4个通道，实际获得{x.shape[1]}个通道")
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            raise ValueError(
                f"期望输入尺寸为{self.input_size}x{self.input_size}，"
                f"实际获得{x.shape[2]}x{x.shape[3]}"
            )

    def train(self, mode=True):
        """重写训练模式切换方法"""
        super().train(mode)
        # 确保CLIP模型始终在评估模式
        if hasattr(self, 'visual_encoder'):
            self.visual_encoder.eval()
        return self

    def initialize_weights(self):
        """初始化模型权重"""
        visual_encoder_modules = list(self.visual_encoder.modules())
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
