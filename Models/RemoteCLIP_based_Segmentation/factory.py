import inspect  
from .architectures.clip_vit_seg import CLIPVITSegmentation  
from .architectures.reclip_rn50_seg import ReCLIPResNetSeg
from .architectures.reclip_rn50_unet_seg import UNetWithReCLIPResNet
from .architectures.reclip_vit_seg import ReCLIPViTSeg  


def segmentation_model_factory(model_type, **kwargs):  
    model_mapping = {  
        'CLIPVITSegmentation': CLIPVITSegmentation,  
        'ReCLIPResNetSeg': ReCLIPResNetSeg,  
        'UNetWithReCLIPResNet': UNetWithReCLIPResNet,  
        'ReCLIPViTSeg': ReCLIPViTSeg,  
    }  
    if model_type not in model_mapping:  
        raise ValueError(f"Unknown model type: {model_type}")  
    
    model_class = model_mapping[model_type]  
    signature = inspect.signature(model_class.__init__)  
    param_names = [  
        k for k in signature.parameters.keys()  
        if k != 'self'  
    ]  
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in param_names}  
    
    # 自动补充缺省参数  
    for name, param in signature.parameters.items():  
        if name == 'self':  
            continue  
        if name not in filtered_kwargs and param.default is not inspect.Parameter.empty:  
            filtered_kwargs[name] = param.default  
        elif name not in filtered_kwargs and param.default is inspect.Parameter.empty:  
            raise TypeError(f'Missing required argument: {name}')  
            
    return model_class(**filtered_kwargs)  