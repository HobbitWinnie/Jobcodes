import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  

class DoubleConv(nn.Module):  
    """增强的双卷积块"""  
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.double_conv(x)  

class AttentionGate(nn.Module):  
    """注意力门控模块"""  
    def __init__(self, F_g, F_l, F_int=None):  
        super().__init__()  
        F_int = F_int or min(F_g, F_l)  
        
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

class UpBlock(nn.Module):  
    """增强的上采样块"""  
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):  
        super().__init__()  
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  
        self.attention = AttentionGate(F_g=in_channels//2, F_l=in_channels//2)  
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)  # in_channels而不是in_channels//2  

    def forward(self, x1, x2=None):  
            """改进的前向传播"""  
            x1 = self.up(x1)  
            
            if x2 is not None:  
                # 处理大小不匹配  
                diff_y = x2.size()[2] - x1.size()[2]  
                diff_x = x2.size()[3] - x1.size()[3]  
                x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,  
                            diff_y // 2, diff_y - diff_y // 2])  
                
                # 确保注意力机制被使用  
                x2 = self.attention(g=x1, x=x2)  
                x = torch.cat([x2, x1], dim=1)  
            else:  
                # 如果没有skip connection，仍然使用注意力  
                x = self.attention(g=x1, x=x1)  
                x = torch.cat([x, x1], dim=1)  
                
            return self.conv(x)  

class RemoteClipUNet(nn.Module):  
    """增强的RemoteClipUNet模型，结合CLIP特征和UNet架构"""  
    def __init__(self,  
                 model_name='ViT-B-32',  
                 ckpt_path=None,  
                 num_classes=9,  
                 dropout_rate=0.2,  
                 use_aux_loss=True,  
                 initial_features=128):  
        super().__init__()  
        self.input_size = 224  
        self.use_aux_loss = use_aux_loss  
        self.dropout_rate = dropout_rate  
        self.num_classes = num_classes  

        # 初始化CLIP模型  
        self._init_clip_model(model_name, ckpt_path)  

        # 特征转换层 - 使用残差连接  
        self.feature_transform = nn.Sequential(  
            nn.Conv2d(512, 512, 1, bias=False),  
            nn.BatchNorm2d(512),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            
            nn.Conv2d(512, 1024, 1, bias=False),  
            nn.BatchNorm2d(1024),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            
            nn.Conv2d(1024, initial_features * 16, 1, bias=False),  
            nn.BatchNorm2d(initial_features * 16),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate)  
        )  

        # 添加空间注意力  
        self.spatial_attention = nn.Sequential(  
            nn.Conv2d(initial_features * 16, initial_features * 16, 1),  
            nn.BatchNorm2d(initial_features * 16),  
            nn.Sigmoid()  
        )  

        # 解码器路径  
        self.decoder_blocks = nn.ModuleList([  
            UpBlock(initial_features * 16, initial_features * 8, dropout_rate),  # 7->14  
            UpBlock(initial_features * 8, initial_features * 4, dropout_rate),   # 14->28  
            UpBlock(initial_features * 4, initial_features * 2, dropout_rate),   # 28->56  
            UpBlock(initial_features * 2, initial_features, dropout_rate),       # 56->112  
            UpBlock(initial_features, initial_features // 2, dropout_rate)       # 112->224  
        ])  

        # 输出层  
        self.final_conv = nn.Sequential(  
            nn.Conv2d(initial_features // 2, initial_features // 2, 3, padding=1, bias=False),  
            nn.BatchNorm2d(initial_features // 2),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            nn.Conv2d(initial_features // 2, num_classes, 1)  
        )  

        # 改进的辅助头  
        if use_aux_loss:  
            self.aux_head = nn.Sequential(  
                DoubleConv(initial_features * 16, initial_features * 8, dropout_rate),  
                nn.Conv2d(initial_features * 8, initial_features * 4, 1),  
                nn.BatchNorm2d(initial_features * 4),  
                nn.ReLU(inplace=True),  
                nn.Dropout2d(dropout_rate),  
                nn.Conv2d(initial_features * 4, num_classes, 1)  
            )  

        self.initialize_weights()  

    def extract_features(self, x):  
        """单独的特征提取方法"""  
        with torch.no_grad():  
            # 确保在评估模式  
            self.visual_encoder.eval()  
            # 使用FP32进行CLIP推理  
            with torch.cuda.amp.autocast(enabled=False):  
                visual_features = self.visual_encoder(x)  
        return visual_features  

    def forward(self, x):  
        """前向传播"""  
        # 输入验证  
        self._validate_input(x)  

        # CLIP特征提取  
        visual_features = self.extract_features(x)  

        # 使用autocast进行后续处理  
        with torch.cuda.amp.autocast():  
            # 特征重塑和转换  
            B, D = visual_features.shape  
            x = visual_features.reshape(B, D, 1, 1)  
            x = self.feature_transform(x)  
            # 将特征图调整到7x7  
            x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)  

            # 应用空间注意力  
            attention_weights = self.spatial_attention(x)  
            x = x * attention_weights  

            # 存储用于辅助损失的特征  
            aux_feature = x.clone() if self.use_aux_loss else None  

            # 解码器路径  
            intermediate_features = []  
            for decoder_block in self.decoder_blocks:  
                x = decoder_block(x)  
                intermediate_features.append(x)  

            # 生成主输出  
            main_output = self.final_conv(intermediate_features[-1])  
            outputs = {'main': main_output}  

            # 生成辅助输出  
            if self.use_aux_loss and aux_feature is not None:  
                aux_output = self.aux_head(aux_feature)  
                outputs['aux'] = F.interpolate(  
                    aux_output,  
                    size=(self.input_size, self.input_size),  
                    mode='bilinear',  
                    align_corners=False  
                )  
               
                # 添加深度监督  
                for idx, feat in enumerate(intermediate_features[:-1]):  
                    aux_out = nn.Conv2d(feat.shape[1], self.num_classes, 1).to(feat.device)(feat)  
                    outputs[f'aux_{idx}'] = F.interpolate(  
                        aux_out,  
                        size=(self.input_size, self.input_size),  
                        mode='bilinear',  
                        align_corners=False  
                    )  
            
            # 确保所有输出都参与计算  
            dummy_sum = sum([output.mean() * 0 for output in outputs.values()])  
            outputs['main'] = outputs['main'] + dummy_sum  

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
    
    def _init_clip_model(self, model_name, ckpt_path):  
        """初始化CLIP模型"""  
        try:  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            model, _, self.preprocess_func = open_clip.create_model_and_transforms(  
                model_name,  
                device=device,  
                pretrained=ckpt_path is None  
            )  

            if ckpt_path:  
                ckpt = torch.load(ckpt_path, map_location=device)  
                model.load_state_dict(ckpt)  

            self.visual_encoder = model.visual.float()  
            self.visual_encoder.eval()  

            # 冻结CLIP参数  
            for param in self.visual_encoder.parameters():  
                param.requires_grad = False  

        except Exception as e:  
            raise RuntimeError(f"CLIP模型加载失败: {str(e)}")  

    def initialize_weights(self):  
        """初始化模型权重"""  
        for m in self.modules():  
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):  
                if m not in self.visual_encoder.modules():  
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                    if m.bias is not None:  
                        nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)