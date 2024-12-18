import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import open_clip  
import logging  

class Attention(nn.Module):  
    def __init__(self, embed_dim):  
        super(Attention, self).__init__()  
        self.query = nn.Linear(embed_dim, embed_dim)  
        self.key = nn.Linear(embed_dim, embed_dim)  
        self.value = nn.Linear(embed_dim, embed_dim)  

    def forward(self, x_visual, x_textual):  
        q = self.query(x_visual)  # [batch_size, num_patches, embed_dim]  
        k = self.key(x_textual)    # [batch_size, num_classes, embed_dim]  
        v = self.value(x_textual)  # [batch_size, num_classes, embed_dim]  

        att_weights = torch.matmul(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)  
        att_weights = F.softmax(att_weights, dim=-1)  

        attended_features = torch.matmul(att_weights, v)  
        return attended_features  

class CLIPVITSegmentation(nn.Module):  
    def __init__(self, model_name, class_names, ckpt_path=None, input_size=224, freeze_clip=True):  
        super(CLIPVITSegmentation, self).__init__()  
        self.input_size = input_size  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self._init_clip_model(model_name, class_names, ckpt_path, freeze_clip)   

    def _init_clip_model(self, model_name, class_names, ckpt_path=None, freeze_clip=False):  
        try:  
            # 加载预训练的 CLIP 模型  
            clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained='openai')  
            clip_model.to(self.device)  # 确保将模型放置于正确的设备  
            self.visual_encoder = clip_model.visual  

            # 修改第一层卷积层以接受 4 通道输入  
            self.visual_encoder.conv1 = nn.Conv2d(  
                in_channels=4,  
                out_channels=self.visual_encoder.conv1.out_channels,  
                kernel_size=self.visual_encoder.conv1.kernel_size,  
                stride=self.visual_encoder.conv1.stride,  
                padding=self.visual_encoder.conv1.padding,  
                bias=self.visual_encoder.conv1.bias  
            )  

            # 初始化新通道的权重  
            with torch.no_grad():  
                original_weights = self.visual_encoder.conv1.weight.clone()  
                self.visual_encoder.conv1.weight[:, :3, :, :] = original_weights[:, :3, :, :]  
                self.visual_encoder.conv1.weight[:, 3:4, :, :] = original_weights[:, :3, :, :].mean(dim=1, keepdim=True)  

            if freeze_clip:  
                for param in clip_model.parameters():  
                    param.requires_grad = False  

            with torch.no_grad():  
                text_tokens = open_clip.tokenize(class_names).to(self.device)  # 确保文本标记在同一设备  
                self.text_features = clip_model.encode_text(text_tokens)  
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)  

            self.visual_embed_dim = self.visual_encoder.conv1.out_channels  
            self.embed_dim = self.text_features.shape[1]  
            if self.visual_embed_dim != self.embed_dim:  
                self.vis_proj = nn.Linear(self.visual_embed_dim, self.embed_dim).to(self.device)  
            
            self.attention = Attention(self.embed_dim).to(self.device)  

            self.decoder = nn.Sequential(  
                nn.ConvTranspose2d(self.embed_dim, 256, kernel_size=4, stride=2, padding=1),  
                nn.BatchNorm2d(256),  
                nn.ReLU(inplace=True),  
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
                nn.BatchNorm2d(128),  
                nn.ReLU(inplace=True),  
                nn.ConvTranspose2d(128, len(class_names), kernel_size=4, stride=2, padding=1)  
            ).to(self.device)  

            if ckpt_path:  
                ckpt = torch.load(ckpt_path, map_location=self.device)  
                if 'state_dict' in ckpt:  
                    ckpt = ckpt['state_dict']  
                self.load_state_dict(ckpt, strict=False)  
        
        except Exception as e:  
            logging.error(f"CLIP 模型加载失败: {str(e)}")  
            raise RuntimeError(f"CLIP 模型加载失败: {str(e)}")  

    def forward(self, images):  
        self._validate_input(images)  

        # 确保将输入移到主设备  
        images = images.to(self.device)  

        x = self.visual_encoder.conv1(images)  
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  

        if hasattr(self, 'vis_proj'):  
            x = self.vis_proj(x)  

        x = x / x.norm(dim=-1, keepdim=True)  
        batch_size = images.shape[0]  
        text_features_repeated = self.text_features.unsqueeze(0).expand(batch_size, -1, -1).to(self.device)  # 确保在同一设备上  

        attended_features = self.attention(x, text_features_repeated)  
        # print("attended_features device:", attended_features.device)  
        # print("self.text_features device:", self.text_features.device)
        logits = torch.matmul(attended_features, self.text_features.t().to(self.device))

        num_patches = attended_features.shape[1]  
        grid_size = int(num_patches ** 0.5)  

        logits = logits.permute(0, 2, 1).view(batch_size, -1, grid_size, grid_size)  
        logits = F.interpolate(logits, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)  
        return logits  
    
    def _validate_input(self, x):  
        if x.dim() != 4:  
            raise ValueError(f"输入张量的维度应为 4，而不是 {x.dim()} 维。")  
        if x.shape[1] != 4:  
            raise ValueError(f"输入张量的通道数应为 4，而不是 {x.shape[1]}。")  

# # 示例用法  
# if __name__ == "__main__":  
#     logging.basicConfig(level=logging.INFO)  # 设置日志级别  
#     class_names = [  
#         'background',  
#         'wheat',  
#         'corn',  
#         'sunflower',  
#         'watermelon',  
#         'tomato',  
#         'sugar beet',  
#         'green onion',  
#         'zucchini'  
#     ]  

#     # 确保将模型存放在一个设备上  
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#     model_name = 'ViT-B-32'  
#     model = CLIPVITSegmentation(model_name, class_names).to(device)  
    
#     # 确保模型使用 DataParallel  
#     model = nn.DataParallel(model, device_ids=[0, 1])  

#     # 创建示例输入并确保在同一个设备上  
#     dummy_images = torch.randn(32, 4, 224, 224).to(device)  
#     output = model(dummy_images)  

#     print("输出尺寸：", output.shape)  # 应为 [32, num_classes, 224, 224]