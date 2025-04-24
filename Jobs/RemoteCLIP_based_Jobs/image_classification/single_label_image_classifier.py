import sys  
sys.path.append('/home/nw/Codes')  

import os  
from Models.RemoteCLIP_based_Classification.single_label.factory import ClassifierFactory
from utils.set_logging import setup_logging
from Loaders.WHURS19_Loader import get_loader

# zero-shot classification parameters
SCENE_CLASSES = [
    'Airport', 
    'Beach', 
    'Bridge', 
    'Commercial', 
    'Desert',
    'Farmland', 
    'footballField', 
    'Forest', 
    'Industrial',
    'Meadow', 
    'Mountain', 
    'Park', 
    'Parking', 
    'Pond', 
    'Port',
    'railwayStation',
    'Residential',
    'River',
    'Viaduct'
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

    classifier = ClassifierFactory.create(
        classifier_type='rf',      # knn, fewshot, svm, rf, zeroshot
        ckpt_path=ckpt_path,  
        model_name=model_name,
        labels=SCENE_CLASSES,
        device_ids=[2,3]         
    )

    train_loader, test_loader = get_loader(
        data_path=train_data_path,
        preprocess=classifier.preprocess_func,
        batch_size=192,
        test_size=0.2
    )

    # 配置/训练分类器
    classifier.train(train_loader,test_loader)  


    # 调用分类器进行分类并保存top-3结果至csv文件
    query_folder_path = '/home/Dataset/nw/GF2_Data/26'   # new data
    output_csv_path = '/home/nw/Codes/tmp/rf.csv'
    classifier.classify_images(query_folder_path, output_csv_path)