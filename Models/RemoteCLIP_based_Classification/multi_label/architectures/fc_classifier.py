import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from ..core.base import BaseCLIPClassifier

class FCClassifier(BaseCLIPClassifier):
    """高效多卡FC分类器，实现端到端分batch训练"""

    def __init__(self, ckpt_path: str, num_labels: int, batch_size: int = 1024, **kwargs):
        super().__init__(ckpt_path, **kwargs)
        self.num_labels = num_labels
        self.batch_size = batch_size
        self._build_model()
        self._init_optimizer()

    def _build_model(self):
        """构建分类头"""
        model = self.model.module if hasattr(self.model, 'module') else self.model  
        self.classifier = nn.Sequential(
            nn.Linear(model.visual.output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_labels)
        ).to(self.main_device).float()
        if len(self.device_ids) > 1:
            self.classifier = torch.nn.DataParallel(self.classifier, device_ids=self.device_ids)

    def _init_optimizer(self):
        """初始化优化器"""
        self.criterion = nn.BCEWithLogitsLoss().to(self.main_device)
        self.optimizer = optim.AdamW(self.classifier.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            total_steps=50  # 后续可覆盖
        )

    def train(self, train_loader, val_loader=None, num_epochs=50, **kwargs):
        """分batch、高效数据并行训练FC头"""
        features, labels = self._extract_feature_label_tensors(train_loader)
        fc_dataset = TensorDataset(features, labels)
        fc_loader = DataLoader(fc_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)

        total_steps = num_epochs * len(fc_loader)
        self.scheduler.total_steps = total_steps

        scaler = torch.cuda.amp.GradScaler(enabled=self.main_device.startswith('cuda'))
        self.classifier.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in fc_loader:
                batch_x = batch_x.to(self.main_device)
                batch_y = batch_y.to(self.main_device)
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.main_device.startswith('cuda')):
                    logits = self.classifier(batch_x)
                    loss = self.criterion(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(fc_dataset)

            # 日志与验证
            if ((epoch + 1) % 5 == 0) or (epoch == num_epochs - 1):
                log_msg = f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f}"
                if val_loader:
                    metrics = self.evaluate(val_loader)
                    log_msg += f" | Val F1: {metrics['f1']:.4f}"
                self.logger.info(log_msg)

    def evaluate(self, data_loader) -> dict:
        """按batch评估"""
        self.classifier.eval()
        features, labels = self._extract_feature_label_tensors(data_loader)
        fc_dataset = TensorDataset(features, labels)
        fc_loader = DataLoader(fc_dataset, batch_size=self.batch_size, num_workers=4)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in fc_loader:
                batch_x = batch_x.to(self.main_device)
                with torch.cuda.amp.autocast(enabled=self.main_device.startswith('cuda')):
                    logits = self.classifier(batch_x)
                    preds = logits.sigmoid().cpu()
                all_preds.append(preds)
                all_labels.append(batch_y)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        return self._calc_metrics(all_labels, all_preds)

    def _format_prediction(self, features: np.ndarray) -> dict:
        """格式化单样本预测"""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.main_device)
        with torch.no_grad():
            logits = self.classifier(features)
            probs = logits.sigmoid().cpu().numpy()[0]
        return {
            'predictions': {f'label_{i}': float(p) for i, p in enumerate(probs)},
            'top_label': int(np.argmax(probs))
        }
