import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from ..core.base import BaseRemoteCLIPSeg

class ReCLIPViTSeg(BaseRemoteCLIPSeg):
    def __init__(  
        self,   
        model_name,   
        ckpt_path=None,   
        num_classes=9,   
        input_size=224,   
        freeze_clip=True,
        in_channels=4,  # 支持自定义输入通道数  
        device_ids=None,  
    ):  
        super().__init__(  
            model_name,  
            num_classes,  
            input_size,  
            ckpt_path,  
            freeze_clip,              
            device_ids,
            in_channels
        )  

        # 对于ViT，emb维度可从transformer.width获得  
        embed_dim = self.encoder.transformer.width  
        self.final_conv = nn.Conv2d(  
            in_channels=embed_dim,  
            out_channels=num_classes,  
            kernel_size=1  
        ).to(self.main_device)  

    def forward(self, x):  
        self._validate_input(x)  
        # x = x.to(self.main_device)  
        x = self._forward_features(x)  # [batch, num_patches+1, emb_dim]  
        x = x[:, 1:, :]  # 去掉CLS  
        batch_size, num_patches, embed_dim = x.size()  
        h = w = int(num_patches ** 0.5)  
        x = x.permute(0, 2, 1).contiguous().view(batch_size, embed_dim, h, w)  
        x = self.final_conv(x)  
        # 输出分辨率一致性  
        if x.shape[-2:] != (self.input_size, self.input_size):  
            x = F.interpolate(  
                x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False  
            )  
        return x  

    def _forward_features(self, x):  
        # Patch Embedding  
        x = self.encoder.conv1(x)  # [batch, emb_dim, grid, grid]  
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [batch, emb, num_patches]  
        x = x.permute(0, 2, 1)  # [batch, num_patches, emb]  
        # 添加 [CLS]  
        cls_token = self.encoder.class_embedding.to(x.dtype).to(self.device) + \
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=self.device)  
        x = torch.cat([cls_token, x], dim=1)  
        x = x + self.encoder.positional_embedding.to(x.dtype).to(self.device)  
        x = self.encoder.ln_pre(x)  
        # ViT主干  
        x = x.permute(1, 0, 2)    # [num_patches+1, batch, emb]  
        x = self.encoder.transformer(x)  
        x = x.permute(1, 0, 2)    # [batch, num_patches+1, emb]  
        return x  