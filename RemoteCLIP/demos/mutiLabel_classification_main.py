import sys  
import os  
from torch.utils.data import DataLoader  

# 添加相关路径  
sys.path.append('/home/nw/Codes/data')  
sys.path.append('/home/nw/Codes/RemoteCLIP/src/image_classification')  

# 导入自定义类  
from MultiLabel_CSV_Loader import MultiLabelCSVLoader  
from remoteclip_multilabel import MultiLabelClassifier  


def train_and_evaluate(classifier, dataloader, num_labels, epoch_num, save_path):  
    best_f1 = -1  
    for epoch in range(1, epoch_num + 1):  
        classifier.train_model(  
            dataloader,   
            num_labels,   
            num_epochs=256,  
            lr=1e-4,   
            loss_type='bce', # 可选的4种loss 'bce', 'focal', 'dice', 'label_smoothing'(暂时不可用).  
            alpha=1,  
            gamma=2  
        )          

        f1, average_precision, roc_auc = classifier.evaluate_model(dataloader)  
        
        if f1 > best_f1:  
            best_f1 = f1  
            classifier.save_model(save_path)  # 保存最佳模型  
        
        print(f"Epoch {epoch}/{epoch_num}, "   
              f"F1: {f1:.6f}, Average Precision: {average_precision:.6f}, "   
              f"ROC-AUC: {roc_auc:.6f}")  
        
    classifier.load_model(save_path, num_labels)  # 恢复最佳模型，增加 num_labels 参数
   


if __name__ == "__main__":  
    ROOT_DIR = '/mnt/d/nw/GF2_Data/MultiLabel_dataset'   
    image_folder_path = os.path.join(ROOT_DIR, 'data')  
    csv_path = os.path.join(ROOT_DIR, 'csv_file/labels_v6.csv')   
    ckpt_path = '/home/nw/Codes/RemoteCLIP/cache/checkpoints/RemoteCLIP-ViT-L-14.pt'  
    query_folder = '/mnt/d/nw/GF2_Data/26'  
    
    output_csv_path = '/mnt/d/nw/GF2_Data/26/result.csv'  
    model_save_path = '/home/nw/Codes/RemoteCLIP/cache/models/best_model.pth'  # 指定模型保存路径  

    # initial classifier  
    classifier = MultiLabelClassifier(ckpt_path=ckpt_path)  

    # load data  
    train_dataset = MultiLabelCSVLoader(csv_path, preprocess_func=classifier.preprocess_func)  
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  

    # Train and evaluate the model  
    num_labels = len(os.listdir(image_folder_path))  # Number of label directories  
    epoch_num=10
    train_and_evaluate(classifier, dataloader, num_labels, epoch_num, model_save_path)  

    # Classify images in a folder  
    classifier.classify_images_from_folder(query_folder, output_csv_path) 

