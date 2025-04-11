from .architectures.clip_knn import KNN 
from .architectures.clip_few_shot import FewShot
from .architectures.clip_rf import RF
from .architectures.clip_svm import SVM
from .architectures.clip_zero_shot import ZeroShot

from pathlib import Path  

class ClassifierFactory:  
    """分类器工厂 (支持扩展多分类方法)"""  
    
    _registry = {  
        "knn": ("KNN", ["n_neighbors"]),  
        "fewshot": ("FewShot", ["num_shots"]),  
        "rf": ("RF", ["n_estimators", "max_depth"]),  
        "svm": ("SVM", ["C", "kernel"]),  
        "zeroshot": ("ZeroShot", ["labels"])  
    }  

    @classmethod  
    def create(  
        cls,  
        method: str,  
        ckpt_path: Path,  
        **kwargs  
    ) :  
        """  
        创建分类器实例  
        
        :param method: 分类方法名称 (小写)  
        :param ckpt_path: 模型检查点路径  
        :param kwargs: 各分类器特有参数  
        :return: 分类器实例  
        """  
        method = method.lower()  
        if method not in cls._registry:  
            available = ", ".join(cls._registry.keys())  
            raise ValueError(f"不支持的方法类型，可用选项: {available}")  

        # 获取目标类及必要参数  
        class_name, required_params = cls._registry[method]  
        module = __import__(f"{class_name.lower()}", fromlist=[class_name])  
        classifier_class = getattr(module, f"{class_name}Classifier")  

        # 参数校验  
        missing = [p for p in required_params if p not in kwargs]  
        if missing:  
            raise ValueError(f"缺失必要参数 {missing} (方法: {method})")  

        try:  
            return classifier_class(ckpt_path=ckpt_path, **kwargs)  
        except Exception as e:  
            logger.error(f"实例化失败 ({method}): {str(e)}")  
            raise  


# 使用示例
from pathlib import Path  

# 创建KNN分类器  
knn_classifier = ClassifierFactory.create(  
    method="knn",  
    ckpt_path=Path("models/clip.pth"),  
    n_neighbors=20  
)  

# 批量处理图像  
knn_classifier.classify_images_in_folder(  
    folder_path=Path("dataset/test_images"),  
    output_csv=Path("results/knn_results.csv")  
)  