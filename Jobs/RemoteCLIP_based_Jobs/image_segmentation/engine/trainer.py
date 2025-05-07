
import torch
import gc
import time
import numpy as np
from torch.cuda.amp import autocast
import logging  
from torch.cuda.amp import GradScaler, autocast  

class Trainer:
    """
    通用支持单/多分支输出，兼容主/辅助分支损失，多主干（ViT/非ViT），训练/验证流程结构简明。
    """
    def __init__(self, model, optimizer, scheduler, criterion, exp_dir, config, aux_weight=0.4):  
        self.model = model  
        self.optimizer = optimizer  
        self.scheduler = scheduler  
        self.criterion = criterion  
        self.exp_dir = exp_dir  
        self.config = config  
        self.scaler = GradScaler()  
        self.best_miou = float('-inf')  
        self.aux_weight = aux_weight

        self.logger = logging.getLogger(self.__class__.__name__)  

        m = self.model.module if hasattr(self.model, 'module') else self.model
        self.use_text = hasattr(m, "model_name") and isinstance(m.model_name, str) and "vit" in m.model_name.lower()
        self.device = m.main_device

    def compute_total_loss(self, outputs, targets):
        """
        自动识别dict or tensor模式，主分支与辅分支加权
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
                loss_info['focal_loss_aux'] = 0.0
                loss_info['dice_loss_aux'] = 0.0
            return loss, loss_info
        else:
            loss, info = self.criterion(outputs, targets)
            info['focal_loss_aux'] = 0.0
            info['dice_loss_aux'] = 0.0
            return loss, info

    def _process_text_batch(self, batch, images):
        """
        输入: batch/图片批量，输出: 规范化text批list或None
        """
        if self.use_text and len(batch) > 2:
            text = batch[2]
            if isinstance(text, str):
                text = [text] * images.shape[0]
            elif isinstance(text, (tuple, list)):
                text = list(text)
            else:
                raise RuntimeError(f"text类型异常: {type(text)}")
            if len(text) != images.shape[0]:
                raise RuntimeError(f"text与图像batch不符: text({len(text)}), images({images.shape[0]})")
            return text
        return None

    def _train_one_batch(self, batch):
        """
        单独训练1个batch，返回loss, loss_info，必要异常保护
        """
        images, masks = batch[0].to(self.device), batch[1].to(self.device)
        text = self._process_text_batch(batch, images)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = self.model(images, text) if text is not None else self.model(images)
            loss, loss_info = self.compute_total_loss(outputs, masks)
        if not torch.isfinite(loss):
            raise ValueError(f"检测到无效loss：{loss.item()}")
        self.scaler.scale(loss).backward()
        if self.config['training'].get('max_grad_norm', 1.0) > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training'].get('max_grad_norm', 1.0))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item(), loss_info

    def train(self, train_loader, val_loader):
        total_epochs = self.config['training']['epochs']
        num_classes = self.config['dataset']['num_classes']

        for epoch in range(total_epochs):
            self.model.train()
            epoch_losses = {
                'loss': 0.0, 'focal_loss': 0.0, 'dice_loss': 0.0,
                'focal_loss_aux': 0.0, 'dice_loss_aux': 0.0, 'batches': 0
            }
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']

            for batch_idx, batch in enumerate(train_loader):
                try:
                    loss, loss_info = self._train_one_batch(batch)
                    epoch_losses['loss'] += loss
                    epoch_losses['focal_loss'] += float(loss_info.get('focal_loss', 0))
                    epoch_losses['dice_loss'] += float(loss_info.get('dice_loss', 0))
                    epoch_losses['focal_loss_aux'] += float(loss_info.get('focal_loss_aux', 0))
                    epoch_losses['dice_loss_aux'] += float(loss_info.get('dice_loss_aux', 0))
                    epoch_losses['batches'] += 1
                except Exception as e:
                    self.logger.error(f"[Epoch {epoch} Batch {batch_idx}] 训练出错: {repr(e)}")
                finally:
                    del batch
                    torch.cuda.empty_cache()
                    gc.collect()

            batches = max(epoch_losses['batches'], 1)
            avg_stat = {k: epoch_losses[k] / batches for k in epoch_losses if k != 'batches'}
            epoch_time = time.time() - epoch_start

            val_metrics = self.validate(val_loader, num_classes)
            self.scheduler.step()

            self.logger.info(
                f"Epoch {epoch + 1}/{total_epochs} [{epoch_time:.2f}s], "
                f"Training: [{avg_stat['loss']:.4f} (Focal: {avg_stat['focal_loss']:.4f}, Dice: {avg_stat['dice_loss']:.4f}, "
                f"AuxFocal: {avg_stat['focal_loss_aux']:.4f}, AuxDice: {avg_stat['dice_loss_aux']:.4f})], "
                f"Validation: [{val_metrics['loss']:.4f} (Focal: {val_metrics['focal_loss']:.4f}, Dice: {val_metrics['dice_loss']:.4f}, "
                f"AuxFocal: {val_metrics.get('focal_loss_aux', 0):.4f}, AuxDice: {val_metrics.get('dice_loss_aux', 0):.4f}), "
                f"Acc = {val_metrics['accuracy']:.4f}, mIoU = {val_metrics['mean_iou']:.4f}], LR: {current_lr:.6f}"
            )

            if val_metrics['mean_iou'] > self.best_miou:
                self.best_miou = val_metrics['mean_iou']
                try:
                    torch.save(self.model.state_dict(), str(self.exp_dir / 'best_model.pth'))
                    self.logger.info(f"保存最佳模型 (mIoU: {self.best_miou:.4f})")
                except Exception as e:
                    self.logger.error(f"模型保存失败: {repr(e)}")

            if current_lr < 1e-7:
                self.logger.info("学习率过小，提前停止训练")
                break

        return self.best_miou

    def _validate_one_batch(self, batch, class_metrics, confusion_matrix, num_classes, val_loss):
        images, masks = batch[0].to(self.device), batch[1].to(self.device)
        text = self._process_text_batch(batch, images)
        with torch.no_grad(), autocast():
            outputs = self.model(images, text) if text is not None else self.model(images)
            loss, loss_info = self.compute_total_loss(outputs, masks)
        val_loss['loss'] += loss.item()
        val_loss['focal_loss'] += float(loss_info.get('focal_loss', 0))
        val_loss['dice_loss'] += float(loss_info.get('dice_loss', 0))
        val_loss['focal_loss_aux'] += float(loss_info.get('focal_loss_aux', 0))
        val_loss['dice_loss_aux'] += float(loss_info.get('dice_loss_aux', 0))
        preds = outputs['main'].argmax(1) if isinstance(outputs, dict) else outputs.argmax(1)
        mask_np = masks.cpu().numpy().flatten()
        pred_np = preds.cpu().numpy().flatten()
        for t, p in zip(mask_np, pred_np):
            if (0 <= t < num_classes) and (0 <= p < num_classes):
                confusion_matrix[t, p] += 1
        for cls in range(num_classes):
            mask_cls = (masks == cls)
            class_metrics[cls]['correct'] += ((preds == cls) & mask_cls).sum().item()
            class_metrics[cls]['total'] += mask_cls.sum().item()

    def validate(self, val_loader, num_classes):
        self.model.eval()
        val_loss = {'loss': 0.0, 'focal_loss': 0.0, 'dice_loss': 0.0, 'focal_loss_aux': 0.0, 'dice_loss_aux': 0.0}
        class_metrics = {i: {'correct': 0, 'total': 0} for i in range(num_classes)}
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        for batch_idx, batch in enumerate(val_loader):
            try:
                self._validate_one_batch(batch, class_metrics, confusion_matrix, num_classes, val_loss)
            except Exception as e:
                self.logger.error(f"[Val Batch {batch_idx}] 验证出错: {repr(e)}")
            finally:
                del batch
                torch.cuda.empty_cache()
                gc.collect()

        num_batches = max(len(val_loader), 1)
        results = {k: val_loss[k] / num_batches for k in val_loss}

        # 逐类与全局指标
        class_performance = {}
        class_ious = []
        for cls in range(num_classes):
            metrics = class_metrics[cls]
            denom = metrics['total']
            accuracy = float(metrics['correct'] / denom) if denom else 0.0
            true_positive = confusion_matrix[cls, cls]
            false_positive = confusion_matrix[:, cls].sum() - true_positive
            false_negative = confusion_matrix[cls, :].sum() - true_positive
            iou = true_positive / (true_positive + false_positive + false_negative + 1e-10)
            precision = true_positive / (true_positive + false_positive + 1e-10)
            recall = true_positive / (true_positive + false_negative + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10) if (precision + recall) else 0.0
            class_performance[cls] = {
                'accuracy': accuracy, 'iou': float(iou), 'precision': float(precision),
                'recall': float(recall), 'f1': float(f1), 'sample_count': int(denom)
            }
            class_ious.append(iou)
        total_correct = sum(m['correct'] for m in class_metrics.values())
        total_samples = sum(m['total'] for m in class_metrics.values())
        global_accuracy = total_correct / total_samples if total_samples else 0.0
        mean_iou = float(np.mean(class_ious)) if class_ious else 0.0

        results.update({
            'accuracy': global_accuracy,
            'mean_iou': mean_iou,
            'class_performance': class_performance,
            'confusion_matrix': confusion_matrix.tolist()
        })
        return results