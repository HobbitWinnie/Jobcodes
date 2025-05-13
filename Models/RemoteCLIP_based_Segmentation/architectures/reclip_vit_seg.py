
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

        # 新增文本投影层  
        self.text_to_visual = nn.Linear(768, 
            self.encoder.transformer.width  # 视觉维度（如768）  
        )  

        # if freeze_clip:  
        #     for param in self.text_to_visual.parameters():  
        #         param.requires_grad = True  

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
        """ 与CLIPVITSegmentation保持一致的维度处理 """  
        enc = self.encoder  
        
        # 分块嵌入 (保持官方实现结构)  
        x = enc.conv1(x)  # [B, D, h, w]  
        B, D, h, w = x.shape  
        x = x.flatten(2).permute(0, 2, 1)  # [B, N, D]  
        
        # CLS token处理 (显式设备同步)  
        cls_token = enc.class_embedding.to(device=x.device, dtype=x.dtype)  
        cls_token = cls_token[None, None, :].expand(B, -1, -1)  # [B,1,D]  
        
        # 拼接并添加位置编码  
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]  
        pos_embed = enc.positional_embedding[:x.size(1)].to(x.device)  
        x += pos_embed[None, :, :]  # 正确广播维度  
        
        # Transformer处理流程  
        x = enc.ln_pre(x)  
        x = x.permute(1, 0, 2)  # [N+1, B, D]  
        x = enc.transformer(x)  
        return x.permute(1, 0, 2)  # [B, N+1, D]  

    # @torch.no_grad()  
    def _build_text_features(self, class_names, device=None):  
        """ 仅在文本特征提取后添加投影 """  
        device = device or self.main_device  
        prompts = [self.prompt_tmpl.format(c) for c in class_names]  
        
        # 原有处理流程保持不变  
        text_tokens = self.tokenizer(prompts).to(device)  
        x = self.token_embedding(text_tokens)  
        x += self.positional_embedding.to(device)  
        x = x.permute(1, 0, 2)  
        text_features = self.text_encode(x)  
        eot_indices = text_tokens.argmax(dim=-1)  
        text_features = text_features[eot_indices, torch.arange(x.size(1)), :]  
        
        # 新增投影层应用（核心修正）  
        projected_text = self.text_to_visual(text_features)  # [K, 768]  
        return projected_text / projected_text.norm(dim=-1, keepdim=True)  