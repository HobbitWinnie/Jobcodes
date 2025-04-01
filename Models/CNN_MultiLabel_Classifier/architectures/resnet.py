import torch  
import torch.nn as nn  
from torchvision.models import resnet101, resnet50, ResNet101_Weights, ResNet50_Weights  
from ..core.base_model import BaseMultiLabelClassifier  

class ResNetClassifier(BaseMultiLabelClassifier):  
    """ResNet-101多标签分类器"""      
    def _build_backbone(self):  
        base_model = resnet101(weights=ResNet101_Weights.DEFAULT)  
        in_features = base_model.fc.in_features  
        base_model.fc = nn.Linear(in_features, self.num_classes)  
        self.model = base_model
        
class ResNet50Classifier(ResNetClassifier):  
    """ResNet-50扩展实现示例"""      
    def _build_backbone(self):  
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)  
        in_features = base_model.fc.in_features  
        base_model.fc = nn.Linear(in_features, self.num_classes)  
        self.model = base_model