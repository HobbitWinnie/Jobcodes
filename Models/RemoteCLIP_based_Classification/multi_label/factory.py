from .architectures.fc_classifier import FCClassifier  
from .architectures.mlknn_classifier import MLKNNClassifier  
from .architectures.svm_classifier import RankSVMClassifier  

class ClassifierFactory:
    """分类器工厂"""
    
    _type_map = {
        'fc': FCClassifier,
        'mlknn': MLKNNClassifier,
        'svm': RankSVMClassifier
    }

    @classmethod
    def create(cls, classifier_type: str, **kwargs):
        """
        创建分类器实例
        :param classifier_type: 分类器类型 (fc/mlknn/svm)
        :param kwargs: 初始化参数 (必须包含ckpt_path)
        """
        if classifier_type.lower() not in cls._type_map:
            raise ValueError(f"Invalid classifier type. Options: {list(cls._type_map.keys())}")
            
        return cls._type_map[classifier_type.lower()](**kwargs)
