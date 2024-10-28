import os
import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
from utils import DiceLoss, calculate_metrics, save_model, load_model

class SimpleModel(nn.Module):
    """用于测试的简单模型"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class TestUtils(unittest.TestCase):
    def setUp(self):
        """测试开始前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def tearDown(self):
        """测试结束后的清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_dice_loss_binary(self):
        """测试二分类的Dice损失"""
        criterion = DiceLoss()
        
        # 创建测试数据
        logits = torch.randn(2, 1, 8, 8)  # 批次大小为2，单通道
        targets = torch.randint(0, 2, (2, 8, 8))  # 二分类标签
        
        # 计算损失
        loss = criterion(logits, targets)
        
        # 验证损失值
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)
        self.assertLessEqual(loss.item(), 1)
    
    def test_dice_loss_multiclass(self):
        """测试多分类的Dice损失"""
        criterion = DiceLoss()
        
        # 创建测试数据
        logits = torch.randn(2, 3, 8, 8)  # 3个类别
        targets = torch.randint(0, 3, (2, 8, 8))  # 3分类标签
        
        # 计算损失
        loss = criterion(logits, targets)
        
        # 验证损失值
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)
        self.assertLessEqual(loss.item(), 1)
    
    def test_calculate_metrics_binary(self):
        """测试二分类的评估指标计算"""
        # 创建测试数据
        outputs = torch.sigmoid(torch.randn(2, 1, 8, 8))
        targets = torch.randint(0, 2, (2, 8, 8))
        
        # 计算指标
        metrics = calculate_metrics(outputs, targets, num_classes=2)
        
        # 验证返回的指标
        self.assertIn('accuracy', metrics)
        self.assertIn('mean_iou', metrics)
        self.assertIn('class_ious', metrics)
        self.assertEqual(len(metrics['class_ious']), 2)
        
        # 验证指标值的范围
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        self.assertGreaterEqual(metrics['mean_iou'], 0)
        self.assertLessEqual(metrics['mean_iou'], 1)
    
    def test_calculate_metrics_multiclass(self):
        """测试多分类的评估指标计算"""
        # 创建测试数据
        outputs = torch.randn(2, 3, 8, 8)  # 3个类别
        targets = torch.randint(0, 3, (2, 8, 8))
        
        # 计算指标
        metrics = calculate_metrics(outputs, targets, num_classes=3)
        
        # 验证返回的指标
        self.assertIn('accuracy', metrics)
        self.assertIn('mean_iou', metrics)
        self.assertIn('class_ious', metrics)
        self.assertEqual(len(metrics['class_ious']), 3)
    
    def test_save_and_load_model(self):
        """测试模型的保存和加载"""
        # 保存模型
        epoch = 10
        loss = 0.5
        save_path = self.temp_dir
        save_model(self.model, self.optimizer, epoch, loss, save_path)
        
        # 验证文件是否存在
        expected_path = Path(save_path) / f'model_epoch_{epoch}.pth'
        self.assertTrue(expected_path.exists())
        
        # 创建新的模型和优化器用于加载
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # 加载模型
        loaded_model, loaded_optimizer, loaded_epoch, loaded_loss = load_model(
            new_model, new_optimizer, str(expected_path)
        )
        
        # 验证加载的值
        self.assertEqual(loaded_epoch, epoch)
        self.assertEqual(loaded_loss, loss)
        
        # 验证模型状态字典是否相同
        for (k1, v1), (k2, v2) in zip(
            self.model.state_dict().items(),
            loaded_model.state_dict().items()
        ):
            self.assertEqual(k1, k2)
            self.assertTrue(torch.equal(v1, v2))

if __name__ == '__main__':
    unittest.main()