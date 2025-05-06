import torch  
from torch.cuda.amp import autocast  
import gc  
import time  
import numpy as np  
from .base_trainer import BaseTrainer

class UnetTrainer(BaseTrainer):  
    """  
    支持多输出分支（如 dict={main, aux}），自动主/辅助分支损失加权的Trainer。  
    仍然兼容单tensor输出。  
    """  
    def __init__(self, model, optimizer, scheduler, criterion_single, exp_dir, config, aux_weight=0.4):  
        """  
        criterion_single: 只用于单分支（如CombinedLoss），不负责加权  
        """  
        super().__init__(model, optimizer, scheduler, criterion_single, exp_dir, config)  
        self.aux_weight = aux_weight  

    def compute_total_loss(self, outputs, targets):  
        """  
        自动识别dict or 纯tensor，主分支与辅分支加权  
        """  
        if isinstance(outputs, dict):  
            loss_main, info_main = self.criterion(outputs['main'], targets)  
            loss = loss_main  
            loss_info = {'focal_loss': info_main['focal_loss'], 'dice_loss': info_main['dice_loss']}  
            if 'aux' in outputs and outputs['aux'] is not None:  
                loss_aux, info_aux = self.criterion(outputs['aux'], targets)  
                loss = loss + self.aux_weight * loss_aux  
                loss_info['focal_loss_aux'] = info_aux['focal_loss']  
                loss_info['dice_loss_aux'] = info_aux['dice_loss']  
            else:  
                loss_info['focal_loss_aux'] = 0  
                loss_info['dice_loss_aux'] = 0  
            return loss, loss_info  
        else:  
            loss, info = self.criterion(outputs, targets)  
            info['focal_loss_aux'] = 0  
            info['dice_loss_aux'] = 0  
            return loss, info  

    def train(self, train_loader, val_loader):  
        total_epochs = self.config['training']['epochs']  
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)  
        num_classes = self.config['dataset']['num_classes']  

        for epoch in range(total_epochs):  
            self.model.train()  
            epoch_loss = epoch_focal_loss = epoch_dice_loss = epoch_focal_loss_aux = epoch_dice_loss_aux = batch_count = 0  
            epoch_start = time.time()  
            current_lr = self.optimizer.param_groups[0]['lr']  

            for batch in train_loader:  
                images, masks = batch[0].to(self.device), batch[1].to(self.device)  
                text = batch[2] if len(batch) > 2 and hasattr(self.model, "model_name") and self.model.model_name == 'Vit' else None  

                self.optimizer.zero_grad(set_to_none=True)  
                with autocast():  
                    if text is not None:  
                        outputs = self.model(images, text)  
                    else:  
                        outputs = self.model(images)  
                    loss, loss_info = self.compute_total_loss(outputs, masks)  
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
                epoch_focal_loss_aux += loss_info.get('focal_loss_aux', 0)  
                epoch_dice_loss_aux += loss_info.get('dice_loss_aux', 0)  
                batch_count += 1  
                del outputs, loss  
                torch.cuda.empty_cache()  

            avg_loss = epoch_loss / batch_count  
            avg_focal_loss = epoch_focal_loss / batch_count  
            avg_dice_loss = epoch_dice_loss / batch_count  
            avg_focal_loss_aux = epoch_focal_loss_aux / batch_count  
            avg_dice_loss_aux = epoch_dice_loss_aux / batch_count  
            epoch_time = time.time() - epoch_start  

            # 验证  
            val_metrics = self.validate(val_loader, num_classes)  

            self.scheduler.step()  
            self.logger.info(  
                f"Epoch {epoch + 1}/{total_epochs} [{epoch_time:.2f}s], "  
                f"Training: [{avg_loss:.4f} (Focal: {avg_focal_loss:.4f}, Dice: {avg_dice_loss:.4f}, AuxFocal: {avg_focal_loss_aux:.4f}, AuxDice: {avg_dice_loss_aux:.4f})], "  
                f"Validation: [{val_metrics['loss']:.4f} (Focal: {val_metrics['focal_loss']:.4f}, Dice: {val_metrics['dice_loss']:.4f}, AuxFocal: {val_metrics.get('focal_loss_aux', 0):.4f}, AuxDice: {val_metrics.get('dice_loss_aux', 0):.4f}), "  
                f"Acc = {val_metrics['accuracy']:.4f}, mIoU = {val_metrics['mean_iou']:.4f}], LR: {current_lr:.6f}"  
            )  

            if val_metrics['mean_iou'] > self.best_miou:  
                self.best_miou = val_metrics['mean_iou']  
                torch.save(self.model.state_dict(), self.exp_dir / 'best_model.pth')  
                self.logger.info(f"保存最佳模型 (mIoU: {self.best_miou:.4f})")  

            if current_lr < 1e-7:  
                self.logger.info("学习率过小，停止训练")  
                break  
            gc.collect()  
        return self.best_miou

    def validate(self, val_loader, num_classes):  
        self.model.eval()  
        val_loss = {'loss':0, 'focal_loss':0, 'dice_loss':0, 'focal_loss_aux':0, 'dice_loss_aux':0}  
        class_metrics = {i: {'correct': 0, 'total': 0} for i in range(num_classes)}  
        confusion_matrix = np.zeros((num_classes, num_classes))  

        with torch.no_grad():  
            for batch in val_loader:  
                images, masks = batch[0].to(self.device), batch[1].to(self.device)  
                text = batch[2] if len(batch) > 2 and hasattr(self.model, "model_name") and self.model.model_name == 'Vit' else None  
                with autocast():  
                    if text is not None:  
                        outputs = self.model(images, text)  
                    else:  
                        outputs = self.model(images)  
                    loss, loss_info = self.compute_total_loss(outputs, masks)  
                val_loss['loss'] += loss.item()  
                val_loss['focal_loss'] += loss_info['focal_loss']  
                val_loss['dice_loss'] += loss_info['dice_loss']  
                val_loss['focal_loss_aux'] += loss_info.get('focal_loss_aux', 0)  
                val_loss['dice_loss_aux'] += loss_info.get('dice_loss_aux', 0)  

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
        results = {  
            'loss': val_loss['loss'] / num_batches,  
            'focal_loss': val_loss['focal_loss'] / num_batches,  
            'dice_loss': val_loss['dice_loss'] / num_batches,  
            'focal_loss_aux': val_loss['focal_loss_aux'] / num_batches,  
            'dice_loss_aux': val_loss['dice_loss_aux'] / num_batches,  
        }  
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

        results.update({  
            'accuracy': accuracy,  
            'mean_iou': mean_iou,  
            'class_performance': class_performance,  
            'confusion_matrix': confusion_matrix.tolist()  
        })  
        return results  