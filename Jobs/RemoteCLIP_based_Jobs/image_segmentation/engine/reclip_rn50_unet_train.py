import numpy as np  
import torch  
from engine.base_trainer import BaseTrainer  

class UNetTrainer(BaseTrainer):  
    def validate(self, val_loader,num_classes):  
        self.model.eval()  
        val_loss = 0  
        loss_components = {'focal_loss': 0, 'dice_loss': 0}  
        class_metrics = {i: {'correct': 0, 'total': 0} for i in range(num_classes)}  
        confusion_matrix = np.zeros((num_classes, num_classes))  

        with torch.no_grad():  
            for batch in val_loader:  
                images, masks = batch[0].to(self.device), batch[1].to(self.device)  
                with torch.cuda.amp.autocast():  
                    outputs = self.model(images)  
                    loss, loss_info = self.criterion(outputs, masks)  

                val_loss += loss.item()  
                loss_components['focal_loss'] += loss_info['focal_loss']  
                loss_components['dice_loss'] += loss_info['dice_loss']  

                preds = outputs['main'].argmax(1) if isinstance(outputs, dict) else outputs.argmax(1)  
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