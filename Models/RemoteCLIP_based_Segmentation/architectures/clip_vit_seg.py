import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from ..core.base import BaseRemoteCLIPSeg  


class CLIPVITSegmentation(BaseRemoteCLIPSeg):  
    def __init__(  
        self,  
        model_name,  
        num_classes=9,  
        input_size=224,  
        ckpt_path=None,  
        freeze_clip=False,  
        in_channels=4,  
        device_ids=None,  
    ):  
        super().__init__(  
            model_name,   
            in_channels,  
            input_size,  
            ckpt_path,   
            freeze_clip,  
            device_ids  
        )  

        # 获取视觉编码器维度  
        self.embed_dim = self.encoder.transformer.width  
        # 文本到视觉的特征投影  
        self.text_to_visual = nn.Linear(768, self.embed_dim)  
        # 最终分类卷积  
        self.final_conv = nn.Conv2d(self.embed_dim, num_classes, kernel_size=1)  
        
    def forward(self, x, text):  
        x = x.to(self.main_device)  
        self._validate_input(x)  
        
        # 视觉特征 [B, N_patches, D]  
        visual_feat = self._forward_features(x)[:, 1:, :]  
        B, num_patches, D = visual_feat.shape  # 直接解包三维特征  
        
        # 文本特征处理（严格维度控制）  
        text_feat = self._forward_text(text)  # 必须返回[B, D_text]  
        text_feat = self.text_to_visual(text_feat)  # [B, D_vis]  
        
        # 维度对齐
        text_feat = text_feat.unsqueeze(1)          # [B, 1, D]  
        text_feat = text_feat.expand(B, num_patches, D)  # 显式指定维度  
        
        # 特征融合  
        fused = visual_feat + text_feat  
        
        # 形状转换（安全方法）  
        H = W = int(num_patches ** 0.5)  
        fused = fused.permute(0, 2, 1).view(B, D, H, W)  
        
        out = self.final_conv(fused)  
        if out.shape[-2:] != (self.input_size, self.input_size):  
            out = F.interpolate(out, size=(self.input_size, self.input_size),   
                               mode='bilinear', align_corners=False)  
        return out  

    def _forward_text(self, text):  
        """ open_clip专用文本处理流程 """  
        # 输入验证  
        if not isinstance(text, list) or len(text) == 0:  
            raise ValueError("输入文本必须为非空列表")  
            
        # 设备同步  
        device = self.main_device  
        
        # 分词处理（open_clip专用方式）  
        text_tokens = self.tokenizer(text).to(device)  # 直接返回[B, 77]张量  
        
        # 嵌入层处理  
        embeddings = self.token_embedding(text_tokens)  # [B, 77, D_text]  
        
        # 位置编码（适配open_clip结构）  
        positional_embeddings = self.positional_embedding.to(device)  # [77, D_text]  
        embeddings += positional_embeddings[None, :77, :]  # 广播到[B, 77, D_text]  
        
        # 文本编码器处理  
        embeddings = embeddings.permute(1, 0, 2)  # [77, B, D_text]  
        text_features = self.text_encode(embeddings)  
        
        # EOT token索引（open_clip规范）  
        eot_pos = text_tokens.argmax(dim=-1)  # 最后一个非零位置即EOT  

        # print(f"Tokenizer类型: {type(self.tokenizer)}")  # 应显示open_clip.tokenizer.SimpleTokenizer  
        # print(f"分词结果示例: {text_tokens[0]}")  # 应显示以49407结尾的序列  
        # print(f"EOT位置: {eot_pos.tolist()}")  # 应全为76（CLIP标准长度77的最后一个位置）  

        return text_features[eot_pos, torch.arange(len(text)), :]  # [B, D_text]  
    

    def _forward_features(self, x):  
        """ViT特征提取流程（维度安全版）"""  
        enc = self.encoder  
        
        # 分块嵌入  
        x = enc.conv1(x)  # [B, D, h, w]  
        B, D, h, w = x.shape  
        x = x.flatten(2).permute(0, 2, 1)  # [B, N, D]  
        
        # CLS token处理（关键修复）  
        cls_token = enc.class_embedding.to(device=x.device, dtype=x.dtype)  # 确保设备/类型一致  
        cls_token = cls_token[None, None, :]  # 形状 [1, 1, D]  
        cls_token = cls_token.expand(B, -1, -1)  # 显式扩展为 [B, 1, D]  
        
        # 拼接特征  
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]  
        
        # 位置编码（维度验证）  
        pos_embed = enc.positional_embedding[:x.size(1)]  # [N+1, D]  
        pos_embed = pos_embed.to(device=x.device, dtype=x.dtype).unsqueeze(0)  # [1, N+1, D]  
        x += pos_embed  
        
        # Transformer处理  
        x = enc.ln_pre(x)  
        x = x.permute(1, 0, 2)  # [N+1, B, D]  
        x = enc.transformer(x)  
        return x.permute(1, 0, 2)  # [B, N+1, D]  