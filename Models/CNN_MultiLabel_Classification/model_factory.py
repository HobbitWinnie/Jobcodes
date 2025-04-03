from .architectures import *  

def create_model(arch, num_classes, multi_gpu=False, device=None):  
    """Model factory with error handling"""  
    model_registry = {  
        'resnet101': ResNetClassifier,  
        'resnet50': ResNet50Classifier,  
        'inceptionv3': InceptionV3Classifier,  
        'densenet201': DenseNet201Classifier,  
        'vgg16': VGG16Classifier  
    }  
    
    arch = arch.lower()  
    if arch not in model_registry:  
        available = ', '.join(model_registry.keys())  
        raise ValueError(f"不支持的模型架构: {arch}，可选: {available}")  
    
    return model_registry[arch](num_classes, multi_gpu, device)  