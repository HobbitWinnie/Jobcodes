import torch.nn as nn  
from torchvision.models import densenet201, DenseNet201_Weights  
from ..core.base_model import BaseMultiLabelClassifier  

class DenseNet201Classifier(BaseMultiLabelClassifier):  
    def _build_backbone(self):  
        base_model = densenet201(weights=DenseNet201_Weights.DEFAULT)  
        in_features = base_model.classifier.in_features  
        base_model.classifier = nn.Linear(in_features, self.num_classes)  
        self.model = base_model
