import os  
import sys  
import torch  
from torch import nn  
from torch.utils.data import DataLoader, DistributedSampler  
import torch.multiprocessing as mp  
import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP  

# 添加相关路径  
sys.path.append('/home/nw/Codes/data')  
sys.path.append('/home/nw/Codes/RemoteCLIP/src/image_classification')  

from MultiLabel_Dataset_Loader import MultiLabelDatasetLoader  
from MultiLabel_CSV_Loader import MultiLabelCSVLoader  
from remoteclip_multilabel_classifier import MultiLabelClassifier  

def setup(rank, world_size):  
    # 初始化进程组  
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  
    torch.cuda.set_device(rank)  

def cleanup():  
    # 销毁进程组  
    dist.destroy_process_group()  

def train_and_evaluate(classifier, dataloader, num_labels, epoch_num, save_path, rank):  
    best_f1 = -1  
    for epoch in range(1, epoch_num + 1):  
        classifier.train_model(  
            dataloader,  
            num_labels,  
            num_epochs=10,  
            lr=1e-4,  
            loss_type='focal',  
            alpha=1,  
            gamma=2  
        )  

        f1, average_precision, roc_auc = classifier.evaluate_model(dataloader)  

        if rank == 0 and f1 > best_f1:  
            best_f1 = f1  
            classifier.save_model(save_path)  

        if rank == 0:  
            print(f"Epoch {epoch}/{epoch_num}, "  
                  f"F1: {f1:.6f}, Average Precision: {average_precision:.6f}, "  
                  f"ROC-AUC: {roc_auc:.6f}")  

    if rank == 0:  
        classifier.load_model(save_path, num_labels)  

def main(rank, world_size, image_folder_path, csv_path, ckpt_path, query_folder, output_csv, model_save_path, batch_size=32, epoch_num=10):  
    setup(rank, world_size)  

    # 初始化分类器  
    classifier = MultiLabelClassifier(ckpt_path=ckpt_path, device=rank)  
    num_labels = len(os.listdir(image_folder_path))  # 标签数量  

    # 在包裹 DDP 前先初始化 fc 层  
    classifier.fc = nn.Linear(classifier.model.visual.output_dim, num_labels).to(rank)  
    
    # 然后包裹模型为 DDP  
    classifier.model = DDP(classifier.model, device_ids=[rank], find_unused_parameters=True)  

    # 加载数据集  
    train_dataset = MultiLabelCSVLoader(csv_path, preprocess_func=classifier.preprocess_func)  
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)  
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)  

    # 训练并评估模型  
    train_and_evaluate(classifier, dataloader, num_labels, epoch_num, model_save_path, rank)  

    if rank == 0:  
        classifier.classify_images_from_folder(query_folder, output_csv)  

    cleanup()  

if __name__ == "__main__":  
    # 设置环境变量  
    os.environ['MASTER_ADDR'] = 'localhost'  # 或者你的主节点IP地址  
    os.environ['MASTER_PORT'] = '12355'  # 建议一个可用端口  

    WORLD_SIZE = 8  # GPU的数量  

    ROOT_DIR = '/mnt/d/nw/GF2_Data/MultiLabel_dataset'  
    image_folder_path = os.path.join(ROOT_DIR, 'data')  
    csv_path = os.path.join(ROOT_DIR, 'csv_file/labels_v6.csv')  
    ckpt_path = '/home/nw/Codes/RemoteCLIP/utils/checkpoints/RemoteCLIP-ViT-L-14.pt'  
    query_folder = '/mnt/d/nw/GF2_Data/26'  
    output_csv_path = '/mnt/d/nw/GF2_Data/26/result.csv'  
    model_save_path = '/home/nw/Codes/RemoteCLIP/cache/best_model.pth'  

    mp.spawn(  
        main,  
        args=(WORLD_SIZE, image_folder_path, csv_path, ckpt_path, query_folder, output_csv_path, model_save_path, 32, 100),  
        nprocs=WORLD_SIZE,  
        join=True  
    )