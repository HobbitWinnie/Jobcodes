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
            in_channels=self.visual_encoder.conv1.out_channels,  # ViT 模型的输出通道数  
            out_channels=num_classes,  
            kernel_size=1  
        )  

    def _init_clip_model(self, model_name, ckpt_path=None, freeze_clip=True):  
        """  
        初始化 CLIP 模型，并修改输入层支持 4 通道数据。  
        """  
        try:  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            # 加载预训练的 CLIP 模型  
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')  

            if ckpt_path:  # 如果提供了检查点路径，则加载自定义权重  
                ckpt = torch.load(ckpt_path, map_location=device)  
                model.load_state_dict(ckpt)  

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

        except Exception as e:  
            print(f"CLIP 模型加载失败: {str(e)}")  
            raise RuntimeError(f"CLIP 模型加载失败: {str(e)}")  

    def forward(self, x):  
        self._validate_input(x)  

        # 提取 CLIP 特征  
        if self.training and not any(param.requires_grad for param in self.visual_encoder.parameters()):  
            # 如果在训练模式且 visual_encoder 被冻结，使用 no_grad  
            with torch.no_grad():  
                x = self.visual_encoder(x)  
        else:  
            x = self.visual_encoder(x)  

        # 映射到分割任务的类别数  
        x = self.final_conv(x)  

        # 确保输出尺寸与输入尺寸一致  
        if x.shape[-2:] != (self.input_size, self.input_size):  
            x = F.interpolate(  
                x,  
                size=(self.input_size, self.input_size),  
                mode='bilinear',  
                align_corners=False  
            )  
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