import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  

class CLIPSegmentation(nn.Module):  
    def __init__(self, model_name, ckpt_path=None, num_classes=9, input_size=224, freeze_clip=True):  
        """  
        使用 ViT-L-14 的 CLIP 模型进行分割任务，支持 4 通道输入。

        Args:  
            model_name (str): CLIP 模型名称（例如 'ViT-L-14'）  
            ckpt_path (str, optional): CLIP 检查点路径。如果为空，则加载预训练权重。  
            num_classes (int): 分割任务的类别数  
            input_size (int): 输入图像的大小  
            freeze_clip (bool): 是否冻结 CLIP 模型的权重  
        """  
        super(CLIPSegmentation, self).__init__()  
        self.input_size = input_size  

        # 初始化 CLIP 模型  
        self._init_clip_model(model_name, ckpt_path, freeze_clip)  

        # 添加最终的卷积层，用于将 CLIP 特征映射到分割类别数  
        self.final_conv = nn.Conv2d(  
            in_channels=self.visual_encoder.transformer.width,  # ViT 模型的嵌入维度  
            out_channels=num_classes,  
            kernel_size=1  
        )  

    def _init_clip_model(self, model_name, ckpt_path=None, freeze_clip=False):  
        """  
        初始化 CLIP 模型，并修改输入层支持 4 通道数据。  
        """  
        try:  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            # 加载预训练的 CLIP 模型  
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')   #             

            self.visual_encoder = model.visual  
            self.visual_encoder.eval()  

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

    def forward(self, x):  
        self._validate_input(x)  
        
        # 获取中间特征  
        x = self._forward_features(x)  # [batch_size, num_patches+1, embed_dim]  
        x = x[:, 1:, :]  # 移除 [CLS] 标记  

        batch_size, num_patches, embed_dim = x.size()  

        # 计算特征图尺寸  
        h = w = int(num_patches ** 0.5)  
        x = x.permute(0, 2, 1).contiguous().view(batch_size, embed_dim, h, w)  # [batch_size, embed_dim, h, w]  
        
        # 经过卷积层  
        x = self.final_conv(x)  
        
        # 如果需要，调整输出尺寸  
        if x.shape[-2:] != (self.input_size, self.input_size):  
            x = F.interpolate(  
                x,  
                size=(self.input_size, self.input_size),  
                mode='bilinear',  
                align_corners=False  
            )  
        return x  
    
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