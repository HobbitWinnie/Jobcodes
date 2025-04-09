import sys  
sys.path.append('/home/nw/Codes')  

import os  
import time  
import torch  
import torch.nn as nn  
import torch.cuda.amp as amp
from sklearn.metrics import f1_score  
from Loaders.MLRSNet_loader import get_dataloaders
from Models.CNN_MultiLabel_Classification.model_factory import create_model


def train_model(model, train_loader, val_loader, MODEL_SAVE_DIR, num_epochs=1000):  
    criterion = nn.BCEWithLogitsLoss()  
    scaler = amp.GradScaler()  # 混合精度训练
    best_f1 = 0  
    history = {'train_loss': [], 'val_f1': []}  

    # 创建保存目录  
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, 'last_checkpoint.pth')  
    
    for epoch in range(num_epochs):  
        epoch_loss = 0  
        start_time = time.time()  
        model.train()  

        # 训练阶段  
        for inputs, labels in train_loader:  
            inputs = inputs.to(model.main_device, non_blocking=True)  
            labels = labels.float().to(model.main_device, non_blocking=True)  
            
            model.optimizer.zero_grad()  

            # 混合精度前向  
            with amp.autocast():  
                outputs = model(inputs)  
                loss = criterion(outputs, labels)  

            # 梯度缩放和反向传播  
            scaler.scale(loss).backward()  
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪  
            scaler.step(model.optimizer)  
            scaler.update()  
            
            epoch_loss += loss.item()  

        # 验证阶段（每5个epoch验证一次）  
        if epoch % 5 == 0:  
            val_f1 = evaluate(model, val_loader)  
            history['val_f1'].append(val_f1)  
            history['train_loss'].append(epoch_loss/len(train_loader))  
            
            # 更新学习率  
            model.scheduler.step(val_f1)  
            
            # 保存最佳模型  
            if val_f1 > best_f1:  
                best_f1 = val_f1  
                torch.save({  
                    'epoch': epoch,  
                    'state_dict': model.state_dict(),  
                    'optimizer': model.optimizer.state_dict(),  
                    'scheduler': model.scheduler.state_dict(),  
                }, os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))  
                
            # 定期保存检查点  
            if (epoch + 1) % 50 == 0:  
                torch.save({  
                    'epoch': epoch,  
                    'state_dict': model.state_dict(),  
                    'optimizer': model.optimizer.state_dict(),  
                    'scheduler': model.scheduler.state_dict(),  
                }, checkpoint_path)  
                
        # 训练日志  
        lr = model.optimizer.param_groups[0]['lr']  
        print(f"Epoch {epoch+1:03d} | "  
              f"Loss: {epoch_loss/len(train_loader):.4f} | "  
              f"LR: {lr:.2e} | "  
              f"F1: {val_f1:.4f} | "  
              f"Time: {time.time()-start_time:.1f}s"  
            )  

def evaluate(model, dataloader, threshold=0.5):  
    model.eval()  
    all_preds, all_labels = [], []  
    
    with torch.no_grad(), amp.autocast():  
        for inputs, labels in dataloader:  
            inputs = inputs.to(model.main_device, non_blocking=True)  
            outputs = model(inputs)  
            
            probs = torch.sigmoid(outputs).cpu()  
            preds = (probs > threshold).int()  
            
            all_preds.extend(preds.numpy())  
            all_labels.extend(labels.int().numpy())  
    
    return f1_score(all_labels, all_preds, average="macro")  


if __name__ == "__main__":  

    # MLRSNetDataset
    DATASET_DIR = '/home/Dataset/nw/Multilabel-Datasets/MLRSNet_dataset'
    MODEL_SAVE_DIR = '/home/nw/Codes/Jobs/CNN_MultiLabel_Classifier/model_save'


    # 初始化模型  
    model = create_model(
        arch='resnet101', 
        num_classes=60, 
        multi_gpu=True, 
        device_ids=[2,3]
    )  

    # 加载数据 
    train_loader, test_loader =  get_dataloaders(
        images_dir = os.path.join(DATASET_DIR, 'Images'),
        labels_dir = os.path.join(DATASET_DIR, 'Labels'),  
        preprocess=model.preprocess,
        batch_size=192,  
        num_workers=8,  # 根据CPU核心数调整  
        pin_memory=True,  
        persistent_workers=True      
    )
   
    # 恢复训练（可选）  
    try:  
        checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR,'last_checkpoint.pth'))  
        model.load_state_dict(checkpoint['state_dict'])  
        model.optimizer.load_state_dict(checkpoint['optimizer'])  
        model.scheduler.load_state_dict(checkpoint['scheduler'])  
        print(f"成功从第{checkpoint['epoch']}个epoch恢复训练")  
    except Exception as e:  
        print("未找到检查点，开始新训练")  

    # 训练模型  
    train_model(model, train_loader, test_loader, MODEL_SAVE_DIR, num_epochs=1000)  