import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  
import numpy as np  

class DoubleConv(nn.Module):  
    """双重卷积块"""  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.double_conv = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )  

    def forward(self, x):  
        return self.double_conv(x)  

class RemoteClipUNet(nn.Module):  
    def __init__(self, model_name='ViT-B-32', ckpt_path=None, num_classes=1, use_aux_loss=True, device=None):  
        super().__init__()  
        self.num_classes = num_classes  
        self.use_aux_loss = use_aux_loss  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  
        
        # 加载CLIP模型  
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)  
        
        if ckpt_path:  
            ckpt = torch.load(ckpt_path, map_location='cpu')  
            self.clip_model.load_state_dict(ckpt)  

        # 获取视觉编码器  
        self.visual_encoder = self.clip_model.visual  
        
        # 冻结参数  
        for param in self.visual_encoder.parameters():  
            param.requires_grad = False  

        # 获取图像尺寸和patch size  
        self.image_size = self.visual_encoder.image_size  
        self.patch_size = self.visual_encoder.patch_size  
        self.grid_size = self.image_size[0] // self.patch_size[0]  

        # 特征融合层  
        self.fusion_conv = DoubleConv(768, 256)  

        # 编码器保存的特征  
        self.encoder_features = []  
        
        # 解码器部分  
        self.decoder1 = DoubleConv(256, 128)  
        self.decoder2 = DoubleConv(128, 64)  
        self.decoder3 = DoubleConv(64, 32)  
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  

        # 输出层  
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)  

        if use_aux_loss:  
            self.aux_head = nn.Sequential(  
                DoubleConv(256, 128),  
                nn.Conv2d(128, num_classes, kernel_size=1)  
            )  

    def reshape_visual_features(self, x):  
        """重塑视觉特征为空间形式"""  
        # 移除类别token  
        x = x[:, 1:, :]  
        # 重塑为空间特征 [B, H*W, C] -> [B, C, H, W]  
        x = x.permute(0, 2, 1)  
        x = x.reshape(x.shape[0], x.shape[1], self.grid_size, self.grid_size)  
        return x  

    def forward(self, x):  
        # 确保输入尺寸正确  
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:  
            x = F.interpolate(x, size=(self.image_size, self.image_size),   
                            mode='bilinear', align_corners=False)  
        
        # 视觉编码器特征提取  
        with torch.no_grad():  
            visual_features = self.visual_encoder(x)  
        
        # 重塑特征  
        x = self.reshape_visual_features(visual_features)  
        
        # 特征融合  
        x = self.fusion_conv(x)  
        
        # 存储辅助损失的特征  
        aux_feature = x if self.use_aux_loss else None  

        # 解码器路径  
        x = self.up1(x)  
        x = self.decoder1(x)  
        
        x = self.up2(x)  
        x = self.decoder2(x)  
        
        x = self.up3(x)  
        x = self.decoder3(x)  

        # 最终输出  
        out = self.final_conv(x)  
        
        # 确保输出尺寸与输入匹配  
        if out.shape[-1] != x.shape[-1] or out.shape[-2] != x.shape[-2]:  
            out = F.interpolate(out, size=(x.shape[-2], x.shape[-1]),   
                              mode='bilinear', align_corners=False)  

        if self.use_aux_loss:  
            aux_out = self.aux_head(aux_feature)  
            aux_out = F.interpolate(aux_out, size=(x.shape[-2], x.shape[-1]),   
                                  mode='bilinear', align_corners=False)  
            return {'main': out, 'aux': aux_out}  
        
        return {'main': out}  

    @torch.no_grad()  
    def preprocess_image(self, image):  
        """预处理图像"""  
        return self.preprocess(image)  

def create_loss_fn(aux_weight=0.4):  
    """创建损失函数"""  
    criterion = nn.CrossEntropyLoss()  
    
    def loss_fn(outputs, target):  
        losses = {}  
        
        # 确保target的形状正确  
        if target.dim() == 3:  
            target = target.unsqueeze(1)  
        
        # 主要损失  
        losses['main'] = criterion(outputs['main'], target)  
        
        # 辅助损失  
        if 'aux' in outputs:  
            losses['aux'] = criterion(outputs['aux'], target)  
            losses['total'] = losses['main'] + aux_weight * losses['aux']  
        else:  
            losses['total'] = losses['main']  
            
        return losses  
    
    return loss_fn 