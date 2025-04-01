import torch.nn as nn  
from torchvision.models import inception_v3, Inception_V3_Weights  
from ..core.base_model import BaseMultiLabelClassifier  

class InceptionV3Classifier(BaseMultiLabelClassifier):  
    """Inception-v3多标签分类器"""      
    @property  
    def input_size(self):  
        return (299, 299)  # 覆盖基类默认尺寸  
    
    def _build_backbone(self):  
        base_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)  
        in_features = base_model.fc.in_features  
        base_model.fc = nn.Linear(in_features, self.num_classes)  
        base_model.aux_logits = False  
        self.model = base_model