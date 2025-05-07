
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  
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
        prompt_tmpl="a satellite image of a {}",  
    ):  
        super().__init__(
            model_name, 
            in_channels,
            input_size,
            ckpt_path, 
            freeze_clip,
            device_ids,  
        )
        self.prompt_tmpl = prompt_tmpl          

    @torch.no_grad()  
    def _build_text_features(self, class_names, device=None):  
        """  
        根据类别名动态生成文本特征  
        """  
        device = device or self.main_device  
        prompts = [self.prompt_tmpl.format(c) for c in class_names]  
        text_tokens = open_clip.tokenizer.tokenize(prompts).to(device)  
        text_features = self.text_encoder(text_tokens)  
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
        return text_features # [num_classes, dim]  

    def forward(self, x, class_names):  
        """  
        x: [B, C, H, W]  
        class_names: 支持list of str 或 list of list-str，  
            - 若为batch统一类别，直接是['wheat','corn',..]  
            - 若为每个样本不同类别，输入如[['wheat', 'corn'], ['wheat', 'tomato']]，返回list  
        """  
        self._validate_input(x)  
        if isinstance(class_names[0], str):  # 全batch统一类别  
            text_features = self._build_text_features(class_names, device=x.device)  # [K, dim]  
            patch_features = self._forward_features(x)  # [B, N+1, dim]  
            patch_features = patch_features[:, 1:, :]   # 去掉CLS  [B, N, dim]  
            B, N, C = patch_features.shape  
            h = w = int(N ** 0.5)  
            patch_features = patch_features.permute(0, 2, 1).reshape(B, C, h, w)  # [B, C, h, w]  
            patch_flat = patch_features.flatten(2).transpose(1, 2)  # [B, HW, C]  
            norm_patch = patch_flat / patch_flat.norm(dim=-1, keepdim=True)  
            norm_text = text_features / text_features.norm(dim=-1, keepdim=True)  
            logits = torch.einsum('bnc,kc->bnk', norm_patch, norm_text)  # [B, HW, K]  
            logits = logits.view(B, h, w, -1).permute(0, 3, 1, 2)      # [B, K, h, w]  
            if (h, w) != (self.input_size, self.input_size):  
                logits = F.interpolate(  
                    logits, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False  
                )  
            return logits  # [B, K, H, W]  
        else:  
            # 每个样本类别不同，逐个处理  
            results = []  
            for i, active_classes in enumerate(class_names):  
                text_feat = self._build_text_features(active_classes, device=x.device)  
                patch_feat = self._forward_features(x[i:i+1])[:, 1:, :]  
                N, C = patch_feat.shape[1:3]  
                h = w = int(N ** 0.5)  
                patch_feat = patch_feat.permute(0, 2, 1).reshape(1, C, h, w)  
                patch_flat = patch_feat.flatten(2).transpose(1, 2)  # [1, HW, C]  
                norm_patch = patch_flat / patch_flat.norm(dim=-1, keepdim=True)  
                norm_text = text_feat / text_feat.norm(dim=-1, keepdim=True)  
                logits = torch.einsum('bnc,kc->bnk', norm_patch, norm_text)  # [1, HW, K]  
                logits = logits.view(1, h, w, -1).permute(0, 3, 1, 2)  
                if (h, w) != (self.input_size, self.input_size):  
                    logits = F.interpolate(  
                        logits, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False  
                    )  
                results.append(logits[0])  # [K, H, W]  
            return results  # list of [K, H, W]  
        
    def _forward_features(self, x):  
        # Patch Embedding  
        enc = self.encoder  
        x = enc.conv1(x)  # [batch, emb_dim, grid, grid]  
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [batch, emb, num_patches]  
        x = x.permute(0, 2, 1)  # [batch, num_patches, emb]  
       
        # 添加 [CLS]  
        cls_token = enc.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1).to(x.device)  
        x = torch.cat([cls_token, x], dim=1)          
        # 位置编码，同样用x.device  
        x = x + enc.positional_embedding.to(x.dtype).to(x.device)  
        x = enc.ln_pre(x) 
        # ViT主干  
        x = x.permute(1, 0, 2)    # [num_patches+1, batch, emb]  
        x = enc.transformer(x)  
        x = x.permute(1, 0, 2)    # [batch, num_patches+1, emb]  
        return x  