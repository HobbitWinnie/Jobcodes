import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  

class CLIPVITSegmentation(nn.Module):  
    def __init__(self, model_name, class_names, ckpt_path=None, input_size=224, freeze_clip=True):  
        """  
        使用 ViT-L-14 的 CLIP 模型进行分割任务，支持 4 通道输入, 并利用文本信息。  

        Args:  
            model_name (str): CLIP 模型名称（例如 'ViT-L-14'）  
            class_names (list of str): 分割类别的名称列表，用于生成文本特征  
            ckpt_path (str, optional): CLIP 检查点路径。如果为空，则加载预训练权重。  
            input_size (int): 输入图像的大小  
            freeze_clip (bool): 是否冻结 CLIP 模型的权重  
        """  
        super(CLIPVITSegmentation, self).__init__()  
        self.input_size = input_size  
        self.class_names = class_names  
        self.num_classes = len(class_names)  

        # 初始化 CLIP 模型  
        self._init_clip_model(model_name, ckpt_path, freeze_clip)  

        # 获取类别的文本特征  
        self.text_features = self._get_text_features(self.class_names)  

    def _init_clip_model(self, model_name, ckpt_path=None, freeze_clip=False):  
        """  
        初始化 CLIP 模型，并修改输入层支持 4 通道数据。  
        """  
        try:  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            # 加载预训练的 CLIP 模型  
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')   #             

            self.visual_encoder = model.visual  
            self.text_encoder = model.encode_text  # 文本编码器  
            # self.visual_encoder.eval()  

            # 修改输入层支持 4 通道  
            # 对于 ViT 模型，输入层是 conv1（Patch Embedding）  
            original_conv1 = self.visual_encoder.conv1  
            self.visual_encoder.conv1 = nn.Conv2d(  
                in_channels=4,  # 修改为 4 通道  
                out_channels=original_conv1.out_channels,  
                kernel_size=original_conv1.kernel_size,  
                stride=original_conv1.stride,  
                padding=original_conv1.padding,  
                bias=original_conv1.bias  
            )  

            # 初始化新通道的权重  
            with torch.no_grad():  
                self.visual_encoder.conv1.weight[:, :3, :, :] = original_conv1.weight  # 复制原始 3 通道权重  
                # 第 4 通道的权重初始化为前三个通道权重的平均值  
                avg_weight = original_conv1.weight[:, :3, :, :].mean(dim=1, keepdim=True)  
                self.visual_encoder.conv1.weight[:, 3:4, :, :] = avg_weight  

            # 冻结 CLIP 模型的权重（可选）  
            if freeze_clip:  
                for param in self.visual_encoder.parameters():  
                    param.requires_grad = False  
                for param in self.text_encoder.parameters():  
                    param.requires_grad = False 
            
            if ckpt_path:  # 如果提供了检查点路径，则加载自定义权重  
                ckpt = torch.load(ckpt_path, map_location=device)  
                # 如果 checkpoint 是包含 'state_dict' 的字典  
                if isinstance(ckpt, dict) and 'state_dict' in ckpt:  
                    ckpt = ckpt['state_dict']  
                # 加载状态字典，允许某些键不存在  
                self.load_state_dict(ckpt, strict=False)  

        except Exception as e:  
            print(f"CLIP 模型加载失败: {str(e)}")  
            raise RuntimeError(f"CLIP 模型加载失败: {str(e)}")  
        
    def _get_text_features(self, class_names):  
        """  
        获取每个类别名称的文本特征。  

        Args:  
            class_names (list of str): 类别名称列表  

        Returns:  
            Tensor: 文本特征，形状为 [num_classes, embed_dim]  
        """  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        with torch.no_grad():  
            # 对类别名称进行 tokenize  
            text_tokens = open_clip.tokenizer.tokenize(class_names).to(device)  
            # 获取文本特征  
            text_features = self.text_encoder(text_tokens)  
            # 标准化  
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
        return text_features  # [num_classes, embed_dim]  

    def forward(self, x):  
        self._validate_input(x)  

        # 获取图像的视觉特征  
        visual_feats = self._forward_features(x)  # [batch_size, num_patches+1, embed_dim]  
        batch_size, num_tokens, embed_dim = visual_feats.size()  

        # 移除 [CLS] 标记，仅保留补丁特征  
        patch_feats = visual_feats[:, 1:, :]  # [batch_size, num_patches, embed_dim]  

        # 标准化视觉特征  
        patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)  # [batch_size, num_patches, embed_dim]  

        # 计算视觉特征与文本特征的相似度  
        # 转置文本特征以便矩阵乘法  
        text_feats = self.text_features.t()  # [embed_dim, num_classes]  

        # 计算相似度  
        logits = torch.matmul(patch_feats, text_feats)  # [batch_size, num_patches, num_classes]  

        # 将 logits 重新调整为特征图形状  
        num_patches = patch_feats.shape[1]  
        h = w = int(num_patches ** 0.5)  
        logits = logits.permute(0, 2, 1).contiguous().view(batch_size, self.num_classes, h, w)  # [batch_size, num_classes, h, w]  

        # 如果需要，调整输出尺寸  
        if logits.shape[-2:] != (self.input_size, self.input_size):  
            logits = F.interpolate(  
                logits,  
                size=(self.input_size, self.input_size),  
                mode='bilinear',  
                align_corners=False  
            )  
        return logits  # [batch_size, num_classes, H, W]  

    def _forward_features(self, x):  
        """  
        自定义的特征提取函数，用于获取每个补丁的特征。  

        Args:  
            x (Tensor): 输入张量，形状为 [batch_size, in_channels, height, width]  

        Returns:  
            Tensor: 特征张量，形状为 [batch_size, num_patches+1, embed_dim]  
        """  
        # Patch Embedding  
        x = self.visual_encoder.conv1(x)  # [batch_size, embed_dim, grid_size, grid_size]  
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch_size, embed_dim, num_patches]  
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, embed_dim]  

        # 添加 [CLS] 标记  
        cls_token = self.visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)  
        x = torch.cat([cls_token, x], dim=1)  # [batch_size, num_patches+1, embed_dim]  

        # 添加位置嵌入  
        x = x + self.visual_encoder.positional_embedding.to(x.dtype)  

        x = self.visual_encoder.ln_pre(x)  

        # 转置为 Transformer 的输入格式 [seq_len, batch_size, embed_dim]  
        x = x.permute(1, 0, 2)  # [num_patches+1, batch_size, embed_dim]  

        # 通过 Transformer 模块  
        x = self.visual_encoder.transformer(x)  

        # 转置回 [batch_size, num_patches+1, embed_dim]  
        x = x.permute(1, 0, 2)  # [batch_size, num_patches+1, embed_dim]  

        return x  

    def _validate_input(self, x):  
        """  
        验证输入数据。  
        """  
        if x.dim() != 4:  
            raise ValueError(f"输入应为 4 维张量，实际维度为 {x.dim()}")  
        if x.shape[1] != 4:  
            raise ValueError(f"期望 4 个通道，实际获得 {x.shape[1]} 个通道")  
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:  
            raise ValueError(  
                f"期望输入尺寸为 {self.input_size}x{self.input_size}，"  
                f"实际获得 {x.shape[2]}x{x.shape[3]}"  
            )  

# model = CLIPSegmentation(model_name='ViT-L-14', num_classes=9, input_size=224)  
# dummy_input = torch.randn(1, 4, 224, 224)  # 批大小为 1，4 通道，224x224 的输入  
# output = model(dummy_input)  
# print("输出形状：", output.shape)