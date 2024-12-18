import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  
import traceback  

class CLIPVITSegmentation(nn.Module):  
    def __init__(self, model_name, ckpt_path=None, num_classes=9, input_size=224, freeze_clip=False):  
        """  
        使用 ViT-L-14 的 CLIP 模型进行分割任务，支持 4 通道输入。  

        Args:  
            model_name (str): CLIP 模型名称（例如 'ViT-L-14'）  
            ckpt_path (str, optional): CLIP 检查点路径。如果为空，则加载预训练权重。  
            num_classes (int): 分割任务的类别数  
            input_size (int): 输入图像的大小  
            freeze_clip (bool): 是否冻结 CLIP 模型的权重  
        """  
        super(CLIPVITSegmentation, self).__init__()  
        self.input_size = input_size  

        # 初始化 CLIP 模型  
        self._init_clip_model(model_name, ckpt_path, freeze_clip)  

        # 添加最终的卷积层，用于将 CLIP 特征映射到分割类别数  
        self.final_conv = nn.Conv2d(  
            in_channels=self.visual_encoder.transformer.width,  # ViT 模型的嵌入维度  
            out_channels=num_classes,  
            kernel_size=1  
        )  

        self.text_to_visual = nn.Linear(768, 1024)  # 768到1024的线性变换  

    def _init_clip_model(self, model_name, ckpt_path=None, freeze_clip=False):  
        """  
        初始化 CLIP 模型，并修改输入层支持 4 通道数据。  
        """  
        try:  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained='openai')  

            if model is None or model.visual is None:  
                raise RuntimeError("CLIP model or visual encoder not correctly instantiated.")  

            # 用于提取文本和视觉编码器  
            self.visual_encoder = model.visual  
            self.text_encoder = model  

            # 打印 visual_encoder 的结构以确认  
            print("Visual Encoder Structure:", self.visual_encoder)  
            
            # 检查是否有参数  
            try:  
                # 用 this to check if parameters are available  
                if next(self.visual_encoder.parameters()) is None:  
                    raise RuntimeError("No parameters found in visual encoder.")  
            except StopIteration:  
                raise RuntimeError("Visual encoder has no parameters.")  
            
            # 添加调试信息  
            if self.visual_encoder is None:  
                raise RuntimeError("Visual encoder not initialized.")  
            try:  
                example_param = next(self.visual_encoder.parameters())  
                print(f"Example parameter: {example_param.size()}")  # 打印第一个参数的大小  
            except StopIteration:  
                raise RuntimeError("Visual encoder has no parameters.")  

            self.text_encoder = model  # 这里将 self.text_encoder 指向整个模型  
            
            self.visual_encoder.eval()  
            # self.text_encoder.eval()  

            # 修改输入层支持 4 通道  
            # original_conv1 = self.visual_encoder.conv1  
            # self.visual_encoder.conv1 = nn.Conv2d(  
            #     in_channels=4,  # 修改为 4 通道  
            #     out_channels=original_conv1.out_channels,  
            #     kernel_size=original_conv1.kernel_size,  
            #     stride=original_conv1.stride,  
            #     padding=original_conv1.padding,  
            #     bias=original_conv1.bias  
            # )  

            # 初始化新通道的权重  
            with torch.no_grad():  
                self.visual_encoder.conv1.weight[:, :3, :, :] = original_conv1.weight  # 复制原始 3 通道权重  
                avg_weight = original_conv1.weight[:, :3, :, :].mean(dim=1, keepdim=True)  
                self.visual_encoder.conv1.weight[:, 3:4, :, :] = avg_weight  

            # 冻结 CLIP 模型的权重（可选）  
            if freeze_clip:  
                for param in self.visual_encoder.parameters():  
                    param.requires_grad = False  
                # for param in self.text_encoder.parameters():  
                #     param.requires_grad = False  
            
            if ckpt_path:  # 如果提供了检查点路径，则加载自定义权重  
                ckpt = torch.load(ckpt_path, map_location=device)  
                if isinstance(ckpt, dict) and 'state_dict' in ckpt:  
                    ckpt = ckpt['state_dict']  
                self.load_state_dict(ckpt, strict=False)  

        except Exception as e:  
            print(f"CLIP 模型加载失败: {str(e)}")  
            raise RuntimeError(f"CLIP 模型加载失败: {str(e)}")  

    def forward(self, x, text):  
        self._validate_input(x)  

        print(f"Visual Encoder Initialized: {self.visual_encoder is not None}")  
        if self.visual_encoder is not None:  
            try:  
                params = list(self.visual_encoder.parameters())  
                print(f"Number of parameters in visual encoder: {len(params)}")  
            except Exception as e:  
                print(f"Error checking parameters: {str(e)}")  

        # 获取中间特征  
        visual_features = self._forward_features(x)  # [batch_size, num_patches+1, 1024]  
        visual_features = visual_features[:, 1:, :]  # 移除 [CLS] 标记  

        # 处理文本输入  
        text_features = self._forward_text(text)  # [batch_size, 768]  
        
        # 使用线性层将文本特征转换为与视觉特征的维度相同  
        text_features = self.text_to_visual(text_features)  # 转换维度到 [batch_size, 1024]  
        
        # 确保 text_features 是 [batch_size, 1, 1024]  
        text_features = text_features.unsqueeze(1)  # 变为 [batch_size, 1, 1024]  

        # 扩展到每个补丁  
        text_features = text_features.expand(-1, visual_features.size(1), -1)  # [batch_size, num_patches, embed_dim]  

        # 结合视觉和文本特征  
        combined_features = visual_features + text_features  # 确保两者的维度相同  

        batch_size, num_patches, embed_dim = combined_features.size()  

        # 计算特征图尺寸  
        h = w = int(num_patches ** 0.5)  
        combined_features = combined_features.permute(0, 2, 1).contiguous().view(batch_size, embed_dim, h, w)  # [batch_size, embed_dim, h, w]  

        # 经过卷积层  
        x = self.final_conv(combined_features)  

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

    def _forward_text(self, text):  
        if self.visual_encoder is None:  
            raise RuntimeError("Visual encoder is not initialized.")  
        
        # 调试打印以检查 parameters  
        try:  
            device = next(self.visual_encoder.parameters()).device  
            print(f"Using device: {device}")  
        except StopIteration:  
            raise RuntimeError("No parameters found in visual encoder. Check initialization.")  
        
        try:  
            if not isinstance(text, list) or len(text) == 0:  
                raise ValueError("Input text must be a non-empty list of strings.")  
            
            text_tokens = open_clip.tokenize(text)  
            if text_tokens is None:  
                raise ValueError("Tokenization failed, received None.")  

            text_tokens = text_tokens.to(next(self.visual_encoder.parameters()).device)  
            text_features = self.text_encoder.encode_text(text_tokens)  
            return text_features  
        
        except Exception as e:  
            print(f"Error during text processing: {str(e)}")  
            traceback.print_exc()  # 打印堆栈信息以获得更多调试信息  
            raise

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
        
# # 初始化模型  
# model = CLIPVITSegmentation(model_name='ViT-L-14', num_classes=9, input_size=224)  

# # 创建虚拟输入  
# dummy_input = torch.randn(1, 4, 224, 224)  # 批大小为 1，4 通道，224x224 的输入  
# dummy_text = ["background corn"]  # 示例文本输入  

# # 进行推理  
# output = model(dummy_input, dummy_text)  

# # 打印输出形状  
# print("输出形状：", output.shape)  
