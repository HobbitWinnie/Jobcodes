from .architectures.clip_vit_seg import CLIPVITSegmentation  
from .architectures.seg_rn50 import CLIPResNetSegmentation
from .architectures.seg_rn50_unet import UNetWithCLIP
from .architectures.seg_vit import CLIPSegmentation  

def create_clip_segmentation(model_type, model_name, num_classes=9, input_size=224, ckpt_path=None, freeze_clip=True):  
    """  
    工厂方法，根据类型返回CLIP分割模型实例。  
    model_type: 'vit' 或 'resnet'  
    """  
    if model_type.lower() == 'vit':  
        return CLIPVITSegmentation(model_name, num_classes, input_size, ckpt_path, freeze_clip)  
    elif model_type.lower() == 'resnet':  
        return CLIPResNetSegmentation(model_name, num_classes, input_size, ckpt_path, freeze_clip)  
    else:  
        raise ValueError(f"未知的model_type: {model_type}")  