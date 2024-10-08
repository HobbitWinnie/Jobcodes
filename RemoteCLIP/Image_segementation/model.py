import torch  
import torch.nn as nn  
import open_clip  

def load_clip_model(ckpt_path, model_name='ViT-L-14', device='cpu'):  
    model, _, preprocess_func = open_clip.create_model_and_transforms(model_name)  
   
    ckpt = torch.load(ckpt_path, map_location='cpu')  
    model.load_state_dict(ckpt)  
    model = model.to(device).eval()  
    
    return model, preprocess_func  

class RemoteCLIPViTSegmentation(nn.Module):  
    def __init__(self, clip_model, num_classes):  
        super(RemoteCLIPViTSegmentation, self).__init__()  
        self.encoder = clip_model.visual  
        
        self.decoder = nn.Sequential(  
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),  # Assuming encoder output for ViT  
            nn.ReLU(inplace=True),  
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  
            nn.ReLU(inplace=True),  
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  
            nn.ReLU(inplace=True),  
            nn.ConvTranspose2d(128, num_classes, kernel_size=2, stride=2)  
        )  
        
    def forward(self, x):  
        x = self.encoder(x.type(self.encoder.dtype))  # Ensure correct type conversion  
        x = x.reshape(x.size(0), -1, 14, 14)  # Reshape if necessary; dimensions depend on ViT output  
        x = self.decoder(x)  
        return x