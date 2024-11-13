import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  
import warnings  

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
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)  

    def forward(self, x1, x2=None):  
        x1 = self.up(x1)  
        
        if x2 is not None:  
            # 处理大小不匹配  
            diff_y = x2.size()[2] - x1.size()[2]  
            diff_x = x2.size()[3] - x1.size()[3]  
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,  
                           diff_y // 2, diff_y - diff_y // 2])  
            
            # 应用注意力机制  
            x2_att = self.attention(g=x1, x=x2)  
            x = torch.cat([x2_att, x1], dim=1)  
        else:  
            x = x1  
            
        return self.conv(x)  

class RemoteClipUNet(nn.Module):  
    """增强的RemoteClipUNet模型，结合CLIP特征和UNet架构"""  
    def __init__(self,  
                 model_name='ViT-B-32',  
                 ckpt_path=None,  
                 num_classes=1,  
                 dropout_rate=0.2,  
                 use_aux_loss=True,  
                 initial_features=64):  
        super().__init__()  
        self.input_size = 224  
        self.grid_size = 7  
        self.use_aux_loss = use_aux_loss  
        self.dropout_rate = dropout_rate  
        
        # 初始化CLIP模型  
        self._init_clip_model(model_name, ckpt_path)  
        
        # 特征转换层  
        self.feature_transform = nn.Sequential(  
            nn.Conv2d(512, 1024, 1),  
            nn.BatchNorm2d(1024),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate),  
            nn.Conv2d(1024, initial_features * 16, 1),  
            nn.BatchNorm2d(initial_features * 16),  
            nn.ReLU(inplace=True),  
            nn.Dropout2d(dropout_rate)  
        )  
        
        # 解码器路径  
        self.up1 = UpBlock(initial_features * 16, initial_features * 8, dropout_rate)  
        self.up2 = UpBlock(initial_features * 8, initial_features * 4, dropout_rate)  
        self.up3 = UpBlock(initial_features * 4, initial_features * 2, dropout_rate)  
        self.up4 = UpBlock(initial_features * 2, initial_features, dropout_rate)  
        self.up_final = UpBlock(initial_features, initial_features // 2, dropout_rate)  
        
        # 输出层  
        self.final_conv = nn.Conv2d(initial_features // 2, num_classes, 1)  
        
        # 辅助头  
        if use_aux_loss:  
            self.aux_head = nn.Sequential(  
                DoubleConv(initial_features * 16, initial_features * 8, dropout_rate),  
                nn.Conv2d(initial_features * 8, num_classes, 1)  
            )  
        
        self.initialize_weights()  

    def _init_clip_model(self, model_name, ckpt_path):  
        """初始化CLIP模型"""  
        try:  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            model, _, self.preprocess_func = open_clip.create_model_and_transforms(  
                model_name,  
                device=device  
            )  
            
            if ckpt_path:  
                ckpt = torch.load(ckpt_path, map_location='cpu')  
                model.load_state_dict(ckpt)  
            
            self.visual_encoder = model.visual.float()  
            
            # 冻结CLIP参数  
            for param in self.visual_encoder.parameters():  
                param.requires_grad = False  
                
        except Exception as e:  
            raise RuntimeError(f"CLIP模型加载失败: {str(e)}")  
    
    def initialize_weights(self):  
        """初始化模型权重"""  
        for m in self.modules():  
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):  
                if m not in self.visual_encoder.modules():  # 跳过CLIP模型的权重  
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                    if m.bias is not None:  
                        nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)  

    @torch.cuda.amp.autocast()  
    def forward(self, x):  
        # 输入验证  
        self._validate_input(x)  
        
        # CLIP特征提取  
        with torch.no_grad():  
            visual_features = self.visual_encoder(x)  
        
        # 特征重塑和转换  
        B, D = visual_features.shape  
        x = visual_features.reshape(B, D, 1, 1)  
        x = self.feature_transform(x)  
        x = F.interpolate(x, size=(self.grid_size, self.grid_size),  
                         mode='bilinear', align_corners=False)  
        
        # 存储辅助特征  
        aux_feature = x if self.use_aux_loss else None  
        
        # 解码器路径  
        x = self.up1(x)  
        x = self.up2(x)  
        x = self.up3(x)  
        x = self.up4(x)  
        x = self.up_final(x)  
        
        # 生成输出  
        outputs = {'main': self.final_conv(x)}  
        
        # 添加辅助输出  
        if self.use_aux_loss and aux_feature is not None:  
            aux_out = self.aux_head(aux_feature)  
            outputs['aux'] = F.interpolate(  
                aux_out,  
                size=(self.input_size, self.input_size),  
                mode='bilinear',  
                align_corners=False  
            )  
        
        return outputs  
    
    def _validate_input(self, x):  
        """验证输入数据"""  
        if x.shape[1] != 3:  
            raise ValueError(f"期望3个通道，实际获得{x.shape[1]}个通道")  
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:  
            raise ValueError(  
                f"期望输入尺寸为{self.input_size}x{self.input_size}，"  
                f"实际获得{x.shape[2]}x{x.shape[3]}"  
            )  

class SegmentationLoss:  
    """分割损失函数"""  
    def __init__(self, aux_weight=0.4, ignore_index=255):  
        self.aux_weight = aux_weight  
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)  
    
    def __call__(self, outputs, target):  
        losses = {'main': self.criterion(outputs['main'], target)}  
        
        if 'aux' in outputs:  
            losses['aux'] = self.criterion(outputs['aux'], target)  
            losses['total'] = losses['main'] + self.aux_weight * losses['aux']  
        else:  
            losses['total'] = losses['main']  
            
        return losses  
