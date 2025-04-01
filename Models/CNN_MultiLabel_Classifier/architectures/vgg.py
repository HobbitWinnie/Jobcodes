import torch.nn as nn  
from torchvision.models import vgg16, VGG16_Weights  
from ..core.base_model import BaseMultiLabelClassifier  


class VGG16Classifier(BaseMultiLabelClassifier):  
    """VGG-16 based classifier"""  

    def _build_backbone(self):  
        base_model = vgg16(weights=VGG16_Weights.DEFAULT)  
        in_features = base_model.classifier[-1].in_features  
        base_model.classifier[-1] = nn.Linear(in_features, self.num_classes)  
        self.model = base_model