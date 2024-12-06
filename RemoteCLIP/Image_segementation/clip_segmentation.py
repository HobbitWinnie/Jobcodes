import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  


class CLIPSegmentation(nn.Module):  
    def __init__(self, model_name, ckpt_path, num_classes, input_size=224):  
        """  
        简化后的分割模型，直接使用 CLIP 提取特征，支持 4 通道输入。  

        Args:  
            model_name (str): CLIP 模型名称  
            ckpt_path (str): CLIP 检查点路径  
            num_classes (int): 分割任务的类别数  
            input_size (int): 输入图像的大小  
        """  
        super(CLIPSegmentation, self).__init__()  
        self.input_size = input_size  

        # 初始化 CLIP 模型  
        self._init_clip_model(model_name, ckpt_path)  

        # 添加最终的卷积层，用于将 CLIP 特征映射到分割类别数  
        self.final_conv = nn.Conv2d(  
            self.visual_encoder.layer4[-1].conv3.out_channels,  
            num_classes,  
            kernel_size=1  
        )  

    def _init_clip_model(self, model_name, ckpt_path):  
        """  
        初始化 CLIP 模型，并修改输入层支持 4 通道数据。  
        """  
        try:  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            model, _, _ = open_clip.create_model_and_transforms(model_name)  

            if ckpt_path:  
                ckpt = torch.load(ckpt_path, map_location=device)  
                model.load_state_dict(ckpt)  

            self.visual_encoder = model.visual  
            self.visual_encoder.eval()  

            # 修改输入层支持 4 通道  
            original_conv1 = self.visual_encoder.conv1  
            self.visual_encoder.conv1 = nn.Conv2d(  
                in_channels=4,  # 修改为 4 通道  
                out_channels=original_conv1.out_channels,  
                kernel_size=original_conv1.kernel_size,  
                stride=original_conv1.stride,  
                padding=original_conv1.padding,  
                bias=False  
            )  

            # 初始化新通道的权重  
            with torch.no_grad():  
                self.visual_encoder.conv1.weight[:, :3, :, :] = original_conv1.weight  # 复制原始 3 通道权重  
                self.visual_encoder.conv1.weight[:, 3:, :, :] = original_conv1.weight[:, :1, :, :]  # 初始化第 4 通道权重  

            # 添加适配层，将通道数从 32 升到 64  
            self.visual_encoder.channel_adapter = nn.Conv2d(  
                in_channels=32,  
                out_channels=64,  
                kernel_size=1,  
                stride=1,  
                padding=0,  
                bias=False  
            )  
            self.visual_encoder.bn_adapter = nn.BatchNorm2d(64)  
                
        except Exception as e:  
            print(f"CLIP 模型加载失败: {str(e)}")  
            raise RuntimeError(f"CLIP 模型加载失败: {str(e)}")  

    def forward(self, x):  
        self._validate_input(x)  
        # print(f"输入张量形状: {x.shape}")  # 打印输入形状  

        # 提取 CLIP 特征  
        with torch.no_grad():  
            x = self.visual_encoder.conv1(x)  
            x = self.visual_encoder.bn1(x)  
            x = self.visual_encoder.act1(x)  

            # 升维适配  
            x = self.visual_encoder.channel_adapter(x)  
            x = self.visual_encoder.bn_adapter(x)  
              
            x = self.visual_encoder.layer1(x)  

            x = self.visual_encoder.layer2(x)  

            x = self.visual_encoder.layer3(x)  

            x = self.visual_encoder.layer4(x)  

        # 映射到分割任务的类别数  
        x = self.final_conv(x)  
        # print(f"final_conv 输出形状: {x.shape}")  

        # 确保输出尺寸与输入尺寸一致  
        if x.shape[-2:] != (self.input_size, self.input_size):  
            x = F.interpolate(  
                x,  
                size=(self.input_size, self.input_size),  
                mode='bilinear',  
                align_corners=False  
            )  
            # print(f"插值后输出形状: {x.shape}")  

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