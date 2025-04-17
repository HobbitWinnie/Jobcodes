import sys  
sys.path.append('/home/nw/Codes')  

import os  
import logging  
from torch.utils.data import DataLoader  
from Loaders.WHURS19_Loader import WHURS19Dataset  
from utils.set_logging import setup_logging


# zero-shot classification parameters
SCENE_17_CLASSES = [
    'airport', 'airport', 'bare_land', 'commercial', 'residential',
    'industrial', 'grassland', 'open_water', 'small_water_body',
    'road', 'railway_area', 'bridge', 'viaduct', 'port', 'woodland', 
    'mining_area','power_station'
]


if __name__ == "__main__":  
    ckpt_path = '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-L-14.pt'  
    model_name = "ViT-L-14"  

    """设置日志配置"""
    setup_logging()
    
    # 数据
    DATA_ROOT_DIR = '/home/Dataset/nw'
    train_dataset = 'Classification/WHU-RS19'
    train_data_path = os.path.join(DATA_ROOT_DIR, train_dataset)

    # few-shot classification parameters
    num_shots = 20
    support_dataset_dir = 'GF2_Data/MultiLabel_dataset/data'
    support_dataset_path = os.path.join(DATA_ROOT_DIR, support_dataset_dir)

    # 配置/训练分类器
    if model_type == 'knn':
        classifier = RemoteCLIPClassifierKNN(ckpt_path=ckpt_path, model_name=model_name)  
        dataset = WHURS19DatasetLoader(
            data_path=train_data_path, 
            preprocess_func = classifier.preprocess_func
            )  
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4
            )
        
        classifier.fit_knn(dataloader, n_neighbors=20)  

    elif model_type == 'svm':
        classifier = RemoteCLIPClassifierSVM(
            ckpt_path=ckpt_path, 
            model_name=model_name
            )  
        dataset = WHURS19DatasetLoader(
            data_path = train_data_path, 
            preprocess_func=classifier.preprocess_func
            )  
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4
            )
        
        classifier.fit_svm(dataloader, C=1.0, kernel='linear')  

    elif model_type == 'rf':
        classifier = RemoteCLIPClassifierRF(
            ckpt_path=ckpt_path, 
            model_name=model_name
            )  
        dataset = WHURS19DatasetLoader(
            data_path=train_data_path, 
            preprocess_func=classifier.preprocess_func
            )  
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4
            )
        
        classifier.fit_rf(dataloader, n_estimators=100, max_depth=None)  

    elif model_type == 'zero-shot':
        labels = SCENE_17_CLASSES
        classifier = RemoteCLIPZeroShotClassifier(
            ckpt_path=ckpt_path, 
            model_name=model_name, 
            labels=labels
            )

    elif model_type == 'few-shot':
        classifier = RemoteCLIPFewShotClassifier(
            ckpt_path=ckpt_path, 
            model_name=model_name
            )
        support_dataset = WHURS19DatasetLoader(
            data_path=support_dataset_path, 
            preprocess_func=classifier.preprocess_func
            )  
        
        classifier.load_support_dataset(support_dataset, num_shots=num_shots)

    else:  
        raise ValueError("Unsupported model type. Choose 'knn', 'zero-shot', 'few-shot', 'svm', or 'rf'.")  


    # 调用分类器进行分类并保存top-3结果至csv文件
    # query_folder_path = '/mnt/d/nw/GF2_Data/data'  
    query_folder_path = '/home/Dataset/nw/GF2_Data/26'   # new data
    output_csv_path = '/home/nw/Codes/results/image_classification/rank_svm_result_v2.csv'
    classifier.classify_images_in_folder(query_folder_path, output_csv_path)