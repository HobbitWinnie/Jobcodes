import inspect  
import torch  
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
    

   # =============== 自动DataParallel包装逻辑 =================  
    device_ids = filtered_kwargs.get("device_ids", None)  
    model = model_class(**filtered_kwargs)  
    
    # 只要device_ids合法且大于1并且cuda可用，就用DataParallel包裹  
    main_device = (f"cuda:{device_ids[0]}" if device_ids else "cuda:0") \
        if torch.cuda.is_available() else "cpu"  
    model.to(main_device)  
    
    if device_ids and len(device_ids) > 1 and torch.cuda.is_available():  
        if hasattr(model, "_logger"):  
            model._logger.info(f"[Factory] Enabled multi-GPU DataParallel on devices: {device_ids}")  
        model = torch.nn.DataParallel(model, device_ids=device_ids)  
    
    return model  