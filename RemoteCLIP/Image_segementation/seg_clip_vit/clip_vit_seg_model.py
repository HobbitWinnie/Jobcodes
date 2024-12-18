import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  
from pathlib import Path  
import logging  
import json  

class CLIPVITSegmentation(nn.Module):  
    def __init__(self, model_name, class_names, ckpt_path=None, input_size=224, freeze_clip=True):  
        super(CLIPVITSegmentation, self).__init__()  

        # 加载 CLIP 模型  
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(  
            model_name, pretrained='openai'  
        )  

        # 定义设备  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        # 将模型移动到设备上  
        self.clip_model.to(self.device)  

        self.input_size = input_size  

        # 获取视觉编码器  
        self.visual_encoder = self.clip_model.visual  

        # **修改第一层卷积层以接受 4 通道输入**  
        self._modify_first_conv_layer()  

        # 根据需要冻结 CLIP 模型参数  
        if freeze_clip:  
            for param in self.clip_model.parameters():  
                param.requires_grad = False  

        # 类别名称  
        self.class_names = class_names  

        # 获取文本特征  
        text_features = self._get_text_features(self.class_names)  # [num_classes, embed_dim]  
        
        # 将文本特征注册为缓冲区  
        self.register_buffer('text_features', text_features) 

        # 获取视觉特征的嵌入维度  
        self.visual_embed_dim = self.visual_encoder.conv1.out_channels  

        # 如果视觉特征和文本特征的维度不匹配，添加一个线性层进行映射  
        self.embed_dim = self.text_features.shape[1]  
        if self.visual_embed_dim != self.embed_dim:  
            self.vis_proj = nn.Linear(self.visual_embed_dim, self.embed_dim).to(self.device)  

        if ckpt_path:  # 如果提供了检查点路径，则加载自定义权重  
            ckpt = torch.load(ckpt_path, map_location=self.device)  
            # 如果 checkpoint 是包含 'state_dict' 的字典  
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:  
                ckpt = ckpt['state_dict']  
            # 加载状态字典，允许某些键不存在  
            self.load_state_dict(ckpt, strict=False)  

    def _modify_first_conv_layer(self):  
        """  
        修改视觉编码器的第一层卷积层，以接受 4 通道输入。  
        """  
        # 获取原始的第一层卷积层  
        orig_conv = self.visual_encoder.conv1  

        # 创建新的卷积层，输入通道数为 4，其他参数与原始层相同  
        new_conv = nn.Conv2d(  
            in_channels=4,  
            out_channels=orig_conv.out_channels,  
            kernel_size=orig_conv.kernel_size,  
            stride=orig_conv.stride,  
            padding=orig_conv.padding,  
            dilation=orig_conv.dilation,  
            groups=orig_conv.groups,  
            bias=orig_conv.bias is not None,  
            padding_mode=orig_conv.padding_mode  
        )  

        # 初始化新的卷积层权重  
        with torch.no_grad():  
            if orig_conv.weight.shape[1] == 3:  
                # 将原始权重复制到新的卷积层的前 3 个通道  
                new_conv.weight[:, :3, :, :] = orig_conv.weight  
                # 第 4 个通道的权重初始化为 3 个通道权重的均值  
                new_conv.weight[:, 3:, :, :] = orig_conv.weight.mean(dim=1, keepdim=True)  
            else:  
                # 如果原始权重不是针对 3 通道的，需要根据具体情况处理  
                raise ValueError("原始卷积层的输入通道数不是 3，无法自动调整。")  

            # 如果有偏置，复制偏置  
            if orig_conv.bias is not None:  
                new_conv.bias = orig_conv.bias  

        # 将新的卷积层替换到视觉编码器中  
        self.visual_encoder.conv1 = new_conv  

    def _get_text_features(self, class_names):  
        """  
        获取每个类别名称的文本特征。  

        Args:  
            class_names (list of str): 类别名称列表  

        Returns:  
            Tensor: 文本特征，形状为 [num_classes, embed_dim]  
        """  

        with torch.no_grad():  
            # 对类别名称进行 tokenize  
            text_tokens = open_clip.tokenizer.tokenize(class_names).to(self.device)  
            # 使用 encode_text 获取文本特征，已经包含投影和标准化  
            text_features = self.clip_model.encode_text(text_tokens)  # [num_classes, embed_dim]  
            # 标准化  
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
        return text_features  # [num_classes, embed_dim]  

    def forward(self, x):  
        # 验证输入  
        self._validate_input(x)  

        # 将输入移动到模型所在的设备上  
        x = x.to(self.device)  

        # 获取视觉特征  
        # with torch.no_grad():  
        # 获取输入嵌入  
        x = self.visual_encoder.conv1(x)  # [batch_size, embed_dim, grid_size, grid_size]  
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch_size, embed_dim, num_patches]  
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, embed_dim]  

        # 如果需要，将视觉特征映射到嵌入维度  
        if hasattr(self, 'vis_proj'):  
            x = self.vis_proj(x)  

        # 标准化视觉特征  
        x = x / x.norm(dim=-1, keepdim=True)  

        # 计算视觉特征与文本特征的相似度  
        logits = torch.matmul(x, self.text_features.t())  # [batch_size, num_patches, num_classes]  

        # 调整形状以形成空间特征图  
        batch_size = x.shape[0]  
        grid_size = int(x.shape[1] ** 0.5)  
        logits = logits.permute(0, 2, 1).reshape(batch_size, -1, grid_size, grid_size)  

        # **上采样 logits 到输入图像的尺寸**  
        logits = F.interpolate(logits, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)  

        return logits  

    def _validate_input(self, x):  
        """  
        验证输入张量的形状和类型。  

        Args:  
            x (Tensor): 输入图像张量  

        Raises:  
            ValueError: 如果输入无效  
        """  
        if x.dim() != 4:  
            raise ValueError(f"输入张量的维度应为 4，而不是 {x.dim()} 维。")  
        if x.shape[1] != 4:  
            raise ValueError(f"输入张量的通道数应为 4，而不是 {x.shape[1]}。")  

# def main():  
#     # 示例主函数，初始化模型并进行一次前向传播  
#     model_name = 'ViT-B-32'  # 您可以根据需要选择模型  
#     class_names = ['猫', '狗', '鸟']  # 示例类别名称  
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

#     # 初始化模型  
#     model = CLIPVITSegmentation(model_name, class_names)  
#     model.to(device)  

#     # 创建示例输入（4 通道）  
#     dummy_input = torch.randn(1, 4, 224, 224).to(device)  

#     # 前向传播  
#     logits = model(dummy_input)  
#     print("输出 logits 形状：", logits.shape)  # 应该是 [batch_size, num_classes, grid_size, grid_size]  

# if __name__ == '__main__':  
#     main()