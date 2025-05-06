import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  
from ..core.base import BaseRemoteCLIPSeg  

class CLIPVITSegmentation(BaseRemoteCLIPSeg):  
    def __init__(
        self,  
        model_name,  
        num_classes=9,  
        input_size=224,  
        ckpt_path=None,  
        freeze_clip=True,  
        in_channels=4,  
        device_ids=None,  
    ):  
        super().__init__(
            model_name, 
            num_classes, 
            input_size, 
            ckpt_path, 
            freeze_clip,
            in_channels,  
            device_ids,  
        )  

        embed_dim = self.encoder.transformer.width  
        self.final_conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1).to(self.main_device)  
        self.text_to_visual = nn.Linear(768, embed_dim).to(self.main_device)  

        # 采用openai公开权重  
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained='openai')  
        self.text_encoder = model.encode_text  
        self.tokenizer = open_clip.get_tokenizer(model_name)  

    def forward(self, x, text):  
        # x = x.to(self.main_device)  
        self._validate_input(x)  
        
        visual_feat = self._forward_features(x)[:, 1:, :]  
        batch_size = x.size(0)  

        if len(text) != batch_size:  
            raise ValueError('文本数量须与图像批次一致')  
        
        text_feat = self._forward_text(text)  
        text_feat = self.text_to_visual(text_feat)  
        num_patches = visual_feat.size(1)  
        embed_dim = visual_feat.size(2)  
        text_feat = text_feat.view(batch_size, 1, embed_dim).expand(-1, num_patches, -1)  

        out = visual_feat + text_feat  
        h = w = int(num_patches ** 0.5)  
        out = out.permute(0, 2, 1).contiguous().view(batch_size, embed_dim, h, w)  
        out = self.final_conv(out)  
        if out.shape[-2:] != (self.input_size, self.input_size):  
            out = F.interpolate(out, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)  
        return out  

    def _forward_features(self, x):  
        """  
        输入x: [B, C, H, W]  
        输出：Transformer每个patch和cls的特征 [B, N_patches+1, embed_dim]  
        """  
        x = self.encoder.conv1(x)   # -> [B, embed, H', W']  
        x = x.reshape(x.shape[0], x.shape[1], -1)                   # [B, embed, PatchNum]  
        x = x.permute(0, 2, 1)                                      # [B, PatchNum, embed_dim]  
        cls_token = self.encoder.class_embedding.to(x.dtype).to(self.main_device) + \
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=self.main_device)  
        x = torch.cat([cls_token, x], dim=1)                        # [B, PatchNum+1, embed_dim]  
        x = x + self.encoder.positional_embedding.to(x.dtype).to(self.main_device)  
        x = self.encoder.ln_pre(x)  
        x = x.permute(1, 0, 2)                                      # [PatchNum+1, B, embed_dim]  
        x = self.encoder.transformer(x)  
        x = x.permute(1, 0, 2)                                      # [B, PatchNum+1, embed_dim]  
        return x  

    def _forward_text(self, text):  
        """  
        文本编码成特征，[B, 768]  
        """  
        if not isinstance(text, list):  
            text = list(text)  
        text_tokens = self.tokenizer(text).to(self.main_device)  
        text_feat = self.text_encoder(text_tokens).to(self.main_device)  
        return text_feat