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
            # 如果尺寸不匹配，调整skip的尺寸  
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

class UNetWithCLIP(nn.Module):  
    def __init__(self, model_name, ckpt_path, num_classes, input_size=224,  
                 dropout_rate=0.1, use_aux_loss=True, initial_features=64):  
        """  
        初始化 UNetWithCLIP 模型  
        
        Args:  
            model_name (str): CLIP 模型名称  
            ckpt_path (str): 检查点路径  
            num_classes (int): 类别数量  
            input_size (int): 输入图像大小  
            dropout_rate (float): dropout 比率  
            use_aux_loss (bool): 是否使用辅助损失  
            initial_features (int): 初始特征通道数  
        """  
        super(UNetWithCLIP, self).__init__()  
        
        self.input_size = input_size  
        self.use_aux_loss = use_aux_loss  
        self.preprocess_func = None
        
        # 初始化 CLIP 模型  
        self._init_clip_model(model_name, ckpt_path)  
        
        # 获取各层输出通道数  
        encoder_layers = self._get_visual_encoder_layers()  
        
        # 特征转换层  
        self.feature_transforms = nn.ModuleList([  
            nn.Conv2d(layer[1], initial_features * (2 ** i), 1)  
            for i, layer in enumerate(encoder_layers)  
        ])  
        
        # 解码器块  
        decoder_in_channels = [initial_features * (2 ** i) for i in range(4, 0, -1)]  
        decoder_skip_channels = [initial_features * (2 ** i) for i in range(3, -1, -1)]  
        decoder_out_channels = [initial_features * (2 ** i) for i in range(3, -1, -1)]  
        
        self.decoder_blocks = nn.ModuleList([  
            DecoderBlock(in_ch, skip_ch, out_ch, dropout_rate)  
            for in_ch, skip_ch, out_ch in zip(  
                decoder_in_channels, decoder_skip_channels, decoder_out_channels  
            )  
        ])  
        
        # 最终卷积层  
        self.final_conv = nn.Sequential(  
            nn.Conv2d(initial_features, initial_features // 2, 3, padding=1, bias=False),  
            nn.BatchNorm2d(initial_features // 2),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            nn.Conv2d(initial_features // 2, num_classes, 1)  
        )  
        
        # 辅助头  
        if use_aux_loss:  
            self.aux_head = nn.Sequential(  
                nn.Conv2d(initial_features * 16, initial_features * 8, 3, padding=1, bias=False),  
                nn.BatchNorm2d(initial_features * 8),  
                nn.ReLU(inplace=True),  
                nn.Dropout2d(dropout_rate),  
                nn.Conv2d(initial_features * 8, num_classes, 1)  
            )  
        
        self.initialize_weights()  

    def _init_clip_model(self, model_name, ckpt_path):  
        """初始化 CLIP 模型"""  
        try:  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            model, _, self.preprocess_func = open_clip.create_model_and_transforms(model_name)  
            
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
        """确保编码器各层通道数一致"""  
        # 保存原始配置  
        original_conv1 = self.visual_encoder.conv1  
        original_bn1 = self.visual_encoder.bn1  
        
        # 重建 conv1  
        self.visual_encoder.conv1 = nn.Conv2d(  
            in_channels=3,  
            out_channels=64,  
            kernel_size=original_conv1.kernel_size,  
            stride=original_conv1.stride,  
            padding=original_conv1.padding,  
            bias=False  
        )  
        
        # 重建 bn1  
        self.visual_encoder.bn1 = nn.BatchNorm2d(  
            num_features=64,  
            eps=original_bn1.eps,  
            momentum=original_bn1.momentum,  
            affine=original_bn1.affine,  
            track_running_stats=original_bn1.track_running_stats  
        )  
        
        # 转换权重（如果需要）  
        if original_conv1.out_channels == 32:  
            with torch.no_grad():  
                original_weights = original_conv1.weight.data  
                new_weights = torch.zeros(64, 3, *original_conv1.kernel_size)  
                new_weights[:32] = original_weights  
                new_weights[32:] = original_weights  
                self.visual_encoder.conv1.weight.data = new_weights  
                
                if original_bn1.affine:  
                    self.visual_encoder.bn1.weight.data[:32] = original_bn1.weight.data  
                    self.visual_encoder.bn1.weight.data[32:] = original_bn1.weight.data  
                    self.visual_encoder.bn1.bias.data[:32] = original_bn1.bias.data  
                    self.visual_encoder.bn1.bias.data[32:] = original_bn1.bias.data  
                
                self.visual_encoder.bn1.running_mean.data[:32] = original_bn1.running_mean.data  
                self.visual_encoder.bn1.running_mean.data[32:] = original_bn1.running_mean.data  
                self.visual_encoder.bn1.running_var.data[:32] = original_bn1.running_var.data  
                self.visual_encoder.bn1.running_var.data[32:] = original_bn1.running_var.data  

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
            
            # 第一层：conv1 + bn1 + act1  
            x = self.visual_encoder.conv1(x)  # 224 -> 112  
            x = self.visual_encoder.bn1(x)  
            x = self.visual_encoder.act1(x)  
            features.append(x)  
            
            # ResNet块  
            x = self.visual_encoder.layer1(x)  # 56 -> 56  
            features.append(x)  
            
            x = self.visual_encoder.layer2(x)  # 56 -> 28  
            features.append(x)  
            
            x = self.visual_encoder.layer3(x)  # 28 -> 14  
            features.append(x)  
            
            x = self.visual_encoder.layer4(x)  # 14 -> 7  
            features.append(x)  
        
        return features  

    def forward(self, x):  
        """前向传播"""  
        self._validate_input(x)  
        
        # 提取CLIP的各层特征  
        features = self.extract_features(x)  
        
        # 对各层特征进行转换  
        features = [  
            transform(f)  
            for transform, f in zip(self.feature_transforms, features)  
        ]  
        
        # 编码器特征用于跳跃连接  
        skips = features[:-1]  
        x = features[-1]  
        
        # 存储辅助特征  
        aux_feature = x.clone() if self.use_aux_loss else None  
        
        # 解码器路径，结合跳跃连接  
        for i, (decoder_block, skip) in enumerate(zip(self.decoder_blocks, reversed(skips))):  
            x = decoder_block(x, skip)  
        
        # 主输出  
        main_output = self.final_conv(x)  
        
        # 确保输出尺寸正确  
        if main_output.shape[-2:] != (self.input_size, self.input_size):  
            main_output = F.interpolate(  
                main_output,  
                size=(self.input_size, self.input_size),  
                mode='bilinear',  
                align_corners=False  
            )  
        
        outputs = {'main': main_output}  
        
        # 辅助输出  
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
        if x.shape[1] != 3:  
            raise ValueError(f"期望3个通道，实际获得{x.shape[1]}个通道")  
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