import torch  
import time  
import numpy as np  
import json  
import logging  
from torch.cuda.amp import GradScaler, autocast  
import gc  

class BaseTrainer:  
    def __init__(self, model, optimizer, scheduler, criterion, exp_dir, config):  
        self.model = model  
        self.optimizer = optimizer  
        self.scheduler = scheduler  
        self.criterion = criterion  
        self.device = model.main_device  
        self.exp_dir = exp_dir  
        self.config = config  
        self.scaler = GradScaler()  
        self.best_miou = float('-inf')  
        self.metrics_history = {  
            'train_loss': [],  
            'train_focal_loss': [],  
            'train_dice_loss': [],  
            'val_loss': [],  
            'val_focal_loss': [],  
            'val_dice_loss': [],  
            'val_miou': [],  
            'val_accuracy': [],  
            'learning_rate': []  
        }  
        self.logger = logging.getLogger(self.__class__.__name__)  

    def train(self, train_loader, val_loader):  
        total_epochs = self.config['training']['epochs']  
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)  
        num_classes = self.config['dataset']['num_classes']  

        for epoch in range(total_epochs):  
            self.model.train()  
            epoch_loss = epoch_focal_loss = epoch_dice_loss = batch_count = 0  
            epoch_start = time.time()  
            current_lr = self.optimizer.param_groups[0]['lr']  

            for batch in train_loader:  
                images, masks = batch[0].to(self.device), batch[1].to(self.device)  
                
                # 针对ViT系列带文本时支持text batch  
                text = batch[2] if len(batch) > 2 and self.model.model_name == 'Vit' else None  

                self.optimizer.zero_grad(set_to_none=True)  
                with autocast():  
                    if text is not None:  
                        outputs = self.model(images, text)  
                    else:  
                        outputs = self.model(images)  
                        
                    loss, loss_info = self.criterion(outputs, masks)  

                if not torch.isfinite(loss):  
                    self.logger.warning(f"检测到非有限损失值: {loss.item()}")  
                    continue  

                self.scaler.scale(loss).backward()                  
                if max_grad_norm > 0:  
                    self.scaler.unscale_(self.optimizer)  
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)  
                self.scaler.step(self.optimizer)  
                self.scaler.update()  

                epoch_loss += loss.item()  
                epoch_focal_loss += loss_info['focal_loss']  
                epoch_dice_loss += loss_info['dice_loss']  
                batch_count += 1  

                del outputs, loss  
                torch.cuda.empty_cache()  

            avg_loss = epoch_loss / batch_count  
            avg_focal_loss = epoch_focal_loss / batch_count  
            avg_dice_loss = epoch_dice_loss / batch_count  
            epoch_time = time.time() - epoch_start  

            self.metrics_history['train_loss'].append(avg_loss)  
            self.metrics_history['train_focal_loss'].append(avg_focal_loss)  
            self.metrics_history['train_dice_loss'].append(avg_dice_loss)  
            self.metrics_history['learning_rate'].append(current_lr)  

            val_metrics = self.validate(val_loader, num_classes)  
            self.metrics_history['val_loss'].append(val_metrics['loss'])  
            self.metrics_history['val_focal_loss'].append(val_metrics['focal_loss'])  
            self.metrics_history['val_dice_loss'].append(val_metrics['dice_loss'])  
            self.metrics_history['val_miou'].append(val_metrics['mean_iou'])  
            self.metrics_history['val_accuracy'].append(val_metrics['accuracy'])  

            self.scheduler.step()  

            self.logger.info(  
                f"Epoch {epoch + 1}/{total_epochs} [{epoch_time:.2f}s], "  
                f"Training: [{avg_loss:.4f} (Focal: {avg_focal_loss:.4f}, Dice: {avg_dice_loss:.4f})], "  
                f"Validation: [{val_metrics['loss']:.4f} (Focal: {val_metrics['focal_loss']:.4f}, Dice: {val_metrics['dice_loss']:.4f}), "  
                f"Acc = {val_metrics['accuracy']:.4f}, mIoU = {val_metrics['mean_iou']:.4f}], LR: {current_lr:.6f}"  
            )  

            if val_metrics['mean_iou'] > self.best_miou:  
                self.best_miou = val_metrics['mean_iou']  
                torch.save(self.model.state_dict(), self.exp_dir / 'best_model.pth')  
                self.logger.info(f"保存最佳模型 (mIoU: {self.best_miou:.4f})")  

            with open(self.exp_dir / 'metrics_history.json', 'w') as f:  
                json.dump(self.metrics_history, f, indent=4)  

            if current_lr < 1e-7:  
                self.logger.info("学习率过小，停止训练")  
                break  
            gc.collect()  
        return self.best_miou, self.metrics_history  

    def validate(self, val_loader, num_classes):  
        self.model.eval()  
        val_loss = 0  
        loss_components = {'focal_loss': 0, 'dice_loss': 0}  
        class_metrics = {i: {'correct': 0, 'total': 0} for i in range(num_classes)}  
        confusion_matrix = np.zeros((num_classes, num_classes))  

        with torch.no_grad():  
            for batch in val_loader:  
                images, masks = batch[0].to(self.device), batch[1].to(self.device)  
                # 针对ViT系列带文本时支持text batch  
                text = batch[2] if len(batch) > 2 and self.model.model_name == 'Vit' else None  

                with autocast():  
                    if text is not None:  
                        outputs = self.model(images, text)  
                    else:  
                        outputs = self.model(images)  
                    loss, loss_info = self.criterion(outputs, masks)  

                val_loss += loss.item()  
                loss_components['focal_loss'] += loss_info['focal_loss']  
                loss_components['dice_loss'] += loss_info['dice_loss']  

                preds = outputs.argmax(1)  
                for true, pred in zip(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten()):  
                    confusion_matrix[true][pred] += 1  
                for cls in range(num_classes):  
                    mask = masks == cls  
                    class_metrics[cls]['correct'] += ((preds == cls) & mask).sum().item()  
                    class_metrics[cls]['total'] += mask.sum().item()  
                del outputs, loss  
                torch.cuda.empty_cache()  

        num_batches = len(val_loader)  
        avg_loss = val_loss / num_batches  
        avg_focal_loss = loss_components['focal_loss'] / num_batches  
        avg_dice_loss = loss_components['dice_loss'] / num_batches  

        class_performance = {}  
        for cls in range(num_classes):  
            metrics = class_metrics[cls]  
            if metrics['total'] > 0:  
                accuracy = metrics['correct'] / metrics['total']  
                true_positive = confusion_matrix[cls][cls]  
                false_positive = confusion_matrix[:, cls].sum() - true_positive  
                false_negative = confusion_matrix[cls, :].sum() - true_positive  
                iou = true_positive / (true_positive + false_positive + false_negative + 1e-10)  
                precision = true_positive / (true_positive + false_positive + 1e-10)  
                recall = true_positive / (true_positive + false_negative + 1e-10)  
                f1 = 2 * precision * recall / (precision + recall + 1e-10)  
                class_performance[cls] = {  
                    'accuracy': float(accuracy),  
                    'iou': float(iou),  
                    'precision': float(precision),  
                    'recall': float(recall),  
                    'f1': float(f1),  
                    'sample_count': int(metrics['total'])  
                }  
        total_correct = sum(m['correct'] for m in class_metrics.values())  
        total_samples = sum(m['total'] for m in class_metrics.values())  
        accuracy = total_correct / total_samples if total_samples > 0 else 0  
        mean_iou = np.mean([p['iou'] for p in class_performance.values()])  

        return {  
            'loss': avg_loss,  
            'focal_loss': avg_focal_loss,  
            'dice_loss': avg_dice_loss,  
            'accuracy': accuracy,  
            'mean_iou': mean_iou,  
            'class_performance': class_performance,  
            'confusion_matrix': confusion_matrix.tolist()  
        }  