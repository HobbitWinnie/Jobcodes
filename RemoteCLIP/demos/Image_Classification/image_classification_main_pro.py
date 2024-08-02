import os  
import sys  
import logging  
from pathlib import Path  
from torch.utils.data import DataLoader  
from PIL import Image  
import shutil 

sys.path.append('/home/nw/Codes/data_loader')  
sys.path.append('/home/nw/Codes/RemoteCLIP/src/image_classification')  

from WHURS19_DatasetLoader import WHURS19DatasetLoader  
from MultiLabel_CSV_Loader import MultiLabelCSVLoader  

from remote_clip_knn import RemoteCLIPClassifierKNN  
from remote_clip_svm import RemoteCLIPClassifierSVM  
from remote_clip_rf import RemoteCLIPClassifierRF  

from remote_clip_zero_shot import RemoteCLIPZeroShotClassifier
from remote_clip_few_shot import RemoteCLIPFewShotClassifier


from remote_clip_ml_knn import RemoteCLIPClassifierMLKNN
from remote_clip_rank_svm import RemoteCLIPClassifierRankSVM

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  


if __name__ == "__main__":  
<<<<<<< HEAD
    ckpt_path = '/home/nw/Codes/RemoteCLIP/cache/checkpoints/RemoteCLIP-ViT-L-14.pt'  
    model_name = "ViT-L-14"  

    # choose a classifier
    model_type = 'rank_svm'

    # train dataset
    data_path ='/mnt/d/nw/Common_Datasets/Classification-12/WHU-RS19'  
    multi_label_data_path = '/mnt/d/nw/GF2_Data/MultiLabel_dataset/data'
=======
    ckpt_path = '/home/nw/Assets/RemoteCLIP/ckpt/RemoteCLIP-ViT-L-14.pt'  
    model_name = "ViT-L-14"  

    # choose a classifier
    model_type = 'svm'

    # train dataset
    data_path ='/home/Dataset/nw/Common_Datasets/Classification-12/WHU-RS19'  
    multi_label_data_path = '/home/Dataset/nw/GF2_Data/MultiLabel_dataset/data'
>>>>>>> 41bc9af (hi)

    # zero-shot classification parameters
    scene_17_labels = [
        'airport', 'airport', 'bare_land', 'commercial', 'residential',
        'industrial', 'grassland', 'open_water', 'small_water_body',
        'road', 'railway_area', 'bridge', 'viaduct', 'port', 'woodland', 
        'mining_area','power_station'
    ]

    # few-shot classification parameters
    num_shots = 20
    support_dataset_path = '/mnt/d/nw/GF2_Data/MultiLabel_dataset/data'

    # multi-label image classification
    ROOT_DIR = '/mnt/d/nw/GF2_Data/MultiLabel_dataset'   
    image_folder_path = os.path.join(ROOT_DIR, 'data')  
    csv_path = os.path.join(ROOT_DIR, 'csv_file/labels_v6.csv')   

    # 配置/训练分类器
    if model_type == 'knn':
        classifier = RemoteCLIPClassifierKNN(ckpt_path=ckpt_path, model_name=model_name)  
        dataset = WHURS19DatasetLoader(data_path=data_path, preprocess_func = classifier.preprocess_func)  
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        classifier.fit_knn(dataloader, n_neighbors=20)  

    elif model_type == 'svm':
        classifier = RemoteCLIPClassifierSVM(ckpt_path=ckpt_path, model_name=model_name)  
        dataset = WHURS19DatasetLoader(data_path=data_path, preprocess_func=classifier.preprocess_func)  
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        classifier.fit_svm(dataloader, C=1.0, kernel='linear')  

    elif model_type == 'rf':
        classifier = RemoteCLIPClassifierRF(ckpt_path=ckpt_path, model_name=model_name)  
        dataset = WHURS19DatasetLoader(data_path=data_path, preprocess_func=classifier.preprocess_func)  
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        classifier.fit_rf(dataloader, n_estimators=100, max_depth=None)  

    elif model_type == 'zero-shot':
        labels = scene_17_labels
        classifier = RemoteCLIPZeroShotClassifier(ckpt_path=ckpt_path, model_name=model_name, labels=labels)

    elif model_type == 'few-shot':
        classifier = RemoteCLIPFewShotClassifier(ckpt_path=ckpt_path, model_name=model_name)
        support_dataset = WHURS19DatasetLoader(data_path=support_dataset_path, preprocess_func=classifier.preprocess_func)  
        classifier.load_support_dataset(support_dataset, num_shots=num_shots)

    # multi-label knn
    elif model_type == 'ml-knn':
        classifier = RemoteCLIPClassifierMLKNN(ckpt_path=ckpt_path, model_name=model_name)  
        dataset = WHURS19DatasetLoader(data_path=multi_label_data_path, preprocess_func = classifier.preprocess_func)  
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        classifier.fit_knn(dataloader, n_neighbors=20)  

    elif model_type == 'rank_svm':
        classifier = RemoteCLIPClassifierRankSVM(ckpt_path=ckpt_path, model_name=model_name)  

        # load data from csv file
        train_dataset = MultiLabelCSVLoader(csv_path, preprocess_func=classifier.preprocess_func)  
        dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
        classifier.fit_rank_svm(dataloader)  

    else:  
        raise ValueError("Unsupported model type. Choose 'knn', 'svm', or 'rf'.")  


    # 调用分类器进行分类并保存top-3结果至csv文件
<<<<<<< HEAD
    # query_folder_path = '/mnt/d/nw/GF2_Data/data'  
    query_folder_path = '/mnt/d/nw/GF2_Data/26'   # new data
    output_csv_path = '/mnt/d/nw/GF2_Data/26/test_rank_svm_top3.csv'
=======
    query_folder_path = '/home/Dataset/nw/GF2_Data/data'  
    # query_folder_path = '/mnt/d/nw/GF2_Data/26'   # new data
    output_csv_path = '/home/Dataset/nw/GF2_Data/26/test_top3.csv'
>>>>>>> 41bc9af (hi)
    classifier.classify_images_in_folder(query_folder_path, output_csv_path)