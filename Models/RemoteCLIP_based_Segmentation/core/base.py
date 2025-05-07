import torch  
import torch.nn as nn  
import open_clip  
import logging  


class BaseRemoteCLIPSeg(nn.Module):  
    """CLIP视觉主干+通用设备/权重多通道适配基础封装"""  
    def __init__(  
        self,  
        model_name,  
        in_channels=4,  
        input_size=224,  
        ckpt_path=None,  
        freeze_clip=True,  
        device_ids=None,  
        logger=None,  
    ):  
        super().__init__()  
        self.model_name = model_name  
        self.in_channels = in_channels  
        self.input_size = input_size  
        self.device_ids = device_ids or []  
        self._logger = logger or logging.getLogger(self.__class__.__name__)  

        # 自动主设备  
        self._validate_devices()  
        self.main_device = self._determine_main_device()  
        self._logger.info(f"Main device: {self.main_device}")    

        # 初始化主干  
        self.visual_encoder, self.text_encoder = self._init_and_patch_encoder(
            model_name, in_channels, ckpt_path, freeze_clip
        )  
        self.encoder_channels = self._get_encoder_channels()  
        
    def _init_and_patch_encoder(self, model_name, in_channels, ckpt_path, freeze_clip):  
        model, _, _ = open_clip.create_model_and_transforms(model_name)  
        model.eval()  
        
        # Patch视觉分支  
        enc = model.visual  
        if hasattr(enc, "layer1") and hasattr(enc, "bn1"):  
            # ----------- ResNet 型主干 -----------  
            target_out_channels = enc.layer1[0].conv1.in_channels  
            new_conv1 = nn.Conv2d(  
                in_channels=in_channels,  
                out_channels=target_out_channels,  
                kernel_size=enc.conv1.kernel_size,  
                stride=enc.conv1.stride,  
                padding=enc.conv1.padding,  
                bias=False)  
            with torch.no_grad():  
                orig = enc.conv1.weight.data  
                num_copy = min(orig.shape[0], new_conv1.weight.shape[0])  
                new_conv1.weight[:num_copy, :3, :, :] = orig[:num_copy]  
                if in_channels > 3:  
                    new_conv1.weight[:num_copy, 3:, :, :] = orig[:num_copy, :1, :, :]  # repeat最后一通道  
            enc.conv1 = new_conv1  

            # Patch BN  
            origbn = enc.bn1  
            enc.bn1 = nn.BatchNorm2d(num_features=target_out_channels, eps=origbn.eps, momentum=origbn.momentum,  
                                    affine=origbn.affine, track_running_stats=origbn.track_running_stats)  
            if origbn.affine:  
                enc.bn1.weight.data[:num_copy] = origbn.weight.data[:num_copy].clone()  
                enc.bn1.bias.data[:num_copy] = origbn.bias.data[:num_copy].clone()  
            enc.bn1.running_mean.data[:num_copy] = origbn.running_mean.data[:num_copy].clone()  
            enc.bn1.running_var.data[:num_copy] = origbn.running_var.data[:num_copy].clone()  
        else:  
            # ----------- ViT 型主干 --------------  
            orig = enc.conv1  
            new_conv = nn.Conv2d(  
                in_channels=in_channels,  
                out_channels=orig.out_channels,  
                kernel_size=orig.kernel_size,  
                stride=orig.stride,  
                padding=orig.padding,  
                bias=orig.bias is not None  
            )  
            with torch.no_grad():  
                # 复制前三通道，后面均值，适配多通道  
                if in_channels == 3:  
                    new_conv.weight[:] = orig.weight  
                else:  
                    new_conv.weight[:, :3] = orig.weight  
                    if in_channels > 3:  
                        mean_weight = orig.weight[:, :3, :, :].mean(dim=1, keepdim=True)  
                        for i in range(3, in_channels):  
                            new_conv.weight[:, i:i+1] = mean_weight  
            enc.conv1 = new_conv  

        # 权重加载（全模型）  
        if ckpt_path:  
            state = torch.load(str(ckpt_path), map_location=self.main_device)  
            if isinstance(state, dict) and "state_dict" in state:  
                state = state['state_dict']  
            model.load_state_dict(state, strict=False)  

        # 冻结clip视觉主干  
        if freeze_clip:  
            for p in enc.parameters():  
                p.requires_grad = False  

        return model.visual, model.encode_text

    def _get_encoder_channels(self):  
        enc = self.encoder  
        # ViT 型  
        if hasattr(enc, 'transformer') and hasattr(enc, 'conv1') and not hasattr(enc, 'layer1'):  
            return [enc.conv1.out_channels]  # ViT 主干仅一层输出（你后续下游可自定义）  
        # ResNet 型  
        elif hasattr(enc, "layer1") and hasattr(enc, "bn1"):  
            return [  
                enc.conv1.out_channels,  
                enc.layer1[-1].conv3.out_channels,  
                enc.layer2[-1].conv3.out_channels,  
                enc.layer3[-1].conv3.out_channels,  
                enc.layer4[-1].conv3.out_channels,  
            ]  
        else:  
            raise NotImplementedError("未知CLIP主干结构")  
    

    # 兼容 DataParallel 封装  
    @property  
    def encoder(self):  
        enc = self.visual_encoder  
        return enc.module if hasattr(enc, "module") else enc  
    
    @property  
    def text_encode(self):  
        tenc = self.text_encoder  
        return tenc if not hasattr(self, 'module') else self.module.text_encoder 
    
    def extract_encoder_features(self, x):  
        enc = self.encoder  
        if hasattr(enc, "layer1") and hasattr(enc, "bn1"):  
            # ResNet特征  
            features = []  
            with torch.no_grad():  
                enc.eval()  
                x = enc.conv1(x)  
                x = enc.bn1(x)  
                x = enc.act1(x)  
                features.append(x)  
                x = enc.layer1(x)  
                features.append(x)  
                x = enc.layer2(x)  
                features.append(x)  
                x = enc.layer3(x)  
                features.append(x)  
                x = enc.layer4(x)  
                features.append(x)  
            return features  
        else:  
            raise NotImplementedError("ViT 型主干请在子类重载 extract_encoder_features")  

    def _validate_input(self, x):  
        if x.dim() != 4:  
            raise ValueError(f"输入应为4维张量，实际为{ x.dim() }维")  
        if x.shape[1] != self.in_channels:  
            raise ValueError(f"期望 {self.in_channels} 通道，实际为{ x.shape[1] }通道")  
        if x.shape[2:] != (self.input_size, self.input_size):  
            raise ValueError(f"期望输入尺寸为{self.input_size}x{self.input_size}，实际为{ x.shape[2] }x{ x.shape[3] }")  
        
    def _determine_main_device(self):  
        if self.device_ids:  
            return f'cuda:{self.device_ids[0]}'
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'  

    def _validate_devices(self):  
        if self.device_ids and not torch.cuda.is_available():  
            raise RuntimeError("指定 device_ids 但 CUDA 不可用")  
        if self.device_ids:  
            available = list(range(torch.cuda.device_count()))  
            for i in self.device_ids:  
                if i not in available:  
                    raise ValueError(f"无效 device_id: {i}, 可选: {available}")  