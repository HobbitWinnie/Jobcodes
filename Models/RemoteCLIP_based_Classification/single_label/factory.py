from .architectures.clip_few_shot import FewShotClassifier 
from .architectures.clip_knn import KNNClassifier
from .architectures.clip_rf import RFClassifier
from .architectures.clip_svm import SVMClassifier
from .architectures.clip_zero_shot import ZeroShotClassifier


class ClassifierFactory:
    """分类器工厂"""
    
    _type_map = {
        'fewshot': FewShotClassifier,  
        'knn': KNNClassifier,  
        'svm': SVMClassifier,  
        'rf': RFClassifier,  
        'zeroshot': ZeroShotClassifier  
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
