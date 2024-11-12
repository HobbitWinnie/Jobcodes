import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import clip  
import warnings  

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

class UpBlock(nn.Module):  
    """上采样块"""  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  
        # 修改：输入通道数是上采样后的通道数  
        self.conv = DoubleConv(in_channels // 2, out_channels)  

    def forward(self, x):  
        x = self.up(x)  
        x = self.conv(x)  
        return x  

class RemoteClipUNet(nn.Module):  
    def __init__(self, model_name='ViT-B/32', ckpt_path=None, num_classes=1, use_aux_loss=True, device=None):  
        super().__init__()  
        self.input_size = 224  
        self.grid_size = 7  
        self.use_aux_loss = use_aux_loss  
        
        # 加载CLIP模型  
        if device is None:  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        try:  
            model, _ = clip.load(model_name, device=device)  
            self.visual_encoder = model.visual.float()  # 添加.float()  
        except Exception as e:  
            warnings.warn(f"Error loading CLIP model: {e}")  
            raise  
        
        # 冻结CLIP参数  
        for param in self.visual_encoder.parameters():  
            param.requires_grad = False  
        
        # 特征转换层  
        self.feature_transform = nn.Sequential(  
            nn.Conv2d(512, 1024, kernel_size=1),  
            nn.BatchNorm2d(1024),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(1024, 768, kernel_size=1),  
            nn.BatchNorm2d(768),  
            nn.ReLU(inplace=True)  
        )  
        
        # 特征融合  
        self.fusion_conv = DoubleConv(768, 512)  
        
        # 解码器路径 - 修改所有上采样块的通道数  
        self.up1 = UpBlock(512, 256)      # 14x14  
        self.up2 = UpBlock(256, 128)      # 28x28  
        self.up3 = UpBlock(128, 64)       # 56x56  
        self.up4 = UpBlock(64, 32)        # 112x112  
        self.up_final = UpBlock(32, 16)   # 224x224  
        
        # 移除多余的解码器块，因为它们已经包含在UpBlock中  
        
        # 最终输出层  
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)  
        
        # 辅助头  
        if use_aux_loss:  
            self.aux_head = nn.Sequential(  
                DoubleConv(512, 256),  
                nn.Conv2d(256, num_classes, kernel_size=1)  
            )  

    def reshape_visual_features(self, x):  
        B, D = x.shape  
        x = x.reshape(B, D, 1, 1)  
        x = self.feature_transform(x)  
        x = F.interpolate(x,   
                         size=(self.grid_size, self.grid_size),  
                         mode='bilinear',  
                         align_corners=False)  
        return x  

    def forward(self, x):  
        # 确保输入尺寸正确  
        assert x.shape[1] == 3, f"Expected 3 channels, got {x.shape[1]}"  
        assert x.shape[2] == self.input_size and x.shape[3] == self.input_size, \
            f"Expected {self.input_size}x{self.input_size} image, got {x.shape[2]}x{x.shape[3]}"  
        
        # 特征提取  
        with torch.no_grad():  
            visual_features = self.visual_encoder(x)  
        
        # 特征处理  
        x = self.reshape_visual_features(visual_features)  
        x = self.fusion_conv(x)  
        
        # 存储辅助损失的特征  
        aux_feature = x if self.use_aux_loss else None  
        
        # 解码器路径  
        x = self.up1(x)        # 14x14  
        x = self.up2(x)        # 28x28  
        x = self.up3(x)        # 56x56  
        x = self.up4(x)        # 112x112  
        x = self.up_final(x)   # 224x224  
        
        # 最终输出  
        out = self.final_conv(x)  
        
        if self.use_aux_loss:  
            aux_out = self.aux_head(aux_feature)  
            aux_out = F.interpolate(aux_out,  
                                  size=(self.input_size, self.input_size),  
                                  mode='bilinear',  
                                  align_corners=False)  
            return {'main': out, 'aux': aux_out}  
        
        return {'main': out}  

def create_loss_fn(aux_weight=0.4, ignore_index=255):  
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)  
    
    def loss_fn(outputs, target):  
        losses = {}  
        losses['main'] = criterion(outputs['main'], target)  
        
        if 'aux' in outputs:  
            losses['aux'] = criterion(outputs['aux'], target)  
            losses['total'] = losses['main'] + aux_weight * losses['aux']  
        else:  
            losses['total'] = losses['main']  
            
        return losses  
    
    return loss_fn  
