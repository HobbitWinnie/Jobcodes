import os  
import sys  
import logging  
from pathlib import Path  
from torch.utils.data import DataLoader  
from PIL import Image  
import shutil 

sys.path.append('/home/nw/Codes/data')  
sys.path.append('/home/nw/Codes/RemoteCLIP/src/image_classification')  

from WHURS19_DatasetLoader import WHURS19DatasetLoader  
from remote_clip_knn import RemoteCLIPClassifierKNN  
from remote_clip_svm import RemoteCLIPClassifierSVM  
from remote_clip_rf import RemoteCLIPClassifierRF  


# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  


if __name__ == "__main__":  
    data_path ='/mnt/d/nw/Common_Datasets/Classification-12/WHU-RS19'  
    ckpt_path = '/home/nw/Codes/RemoteCLIP/cache/checkpoints/RemoteCLIP-ViT-L-14.pt'  

    # 调用分类函数进行分类  
    query_folder_path = '/mnt/d/nw/GF2_Data/data'  
    output_csv_path = '/mnt/d/nw/GF2_Data/data/test_svm_top3.csv'

    model_type = 'svm'

    if model_type == 'knn':
        classifier = RemoteCLIPClassifierKNN(ckpt_path=ckpt_path)  
        dataset = WHURS19DatasetLoader(data_path=data_path, preprocess_func=classifier.preprocess_func)  
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        classifier.fit_knn(dataloader, n_neighbors=20)  

    elif model_type == 'svm':
        classifier = RemoteCLIPClassifierSVM(ckpt_path=ckpt_path)  
        dataset = WHURS19DatasetLoader(data_path=data_path, preprocess_func=classifier.preprocess_func)  
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        classifier.fit_svm(dataloader, C=1.0, kernel='linear')  

    elif model_type == 'rf':
        classifier = RemoteCLIPClassifierRF(ckpt_path=ckpt_path)  
        dataset = WHURS19DatasetLoader(data_path=data_path, preprocess_func=classifier.preprocess_func)  
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        classifier.fit_rf(dataloader, n_estimators=100, max_depth=None)  
    else:  
        raise ValueError("Unsupported model type. Choose 'knn', 'svm', or 'rf'.")  
    
    classifier.classify_images_in_folder(query_folder_path, output_csv_path)