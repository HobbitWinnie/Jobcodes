import json
from pathlib import Path

class Config:
    def __init__(self):
        # 目录与路径
        self.DATA_ROOT = "/home/Dataset/nw/spectraLib/unify_v3"
        self.RESULT_DIR = "/home/nw/Codes/Jobs/SpectraLib_band_selection/result"

        # 波段选择
        self.TOP_N = 5
        self.BAND_COMB_DIM = 10

        # 评价函数
        self.METRIC_WEIGHTS = (0.5, 0.5, 0.0)
        self.ALPHA = 1.0
        self.BETA = 1.0

        # 深度模型参数
        self.VAE_LATENT_DIM = 8
        self.VAE_EPOCHS = 80
        self.VAE_BATCH_SIZE = 16

        self.TRANS_EMBED_DIM = 64
        self.TRANS_DEPTH = 3
        self.TRANS_EPOCHS = 80
        self.TRANS_BATCH_SIZE = 16

        # CLIP参数
        self.CLIP_TOPK = 1
        self.CLIP_THRESH = 0.75

        # 分类/目标
        self.DEFAULT_TARGET_LIST = ["水稻", "草地", "菜地", '玉米', '建筑', '裸地', '建设用地', '道路', '自然植被', '草地', '高速公路', '塑料大棚']

        # 设备与其他
        self.DEVICE = "cuda:1"
        self.SEED = 42

    def update(self, **kwargs):
        """便于脚本/命令行快速修改参数：cfg.update(TOP_N=10, DEVICE='cpu')"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def to_dict(self):
        """返回所有参数（便于保存到文件或传递）"""
        return {k: getattr(self, k) for k in self.__dict__ if not k.startswith('__') and not callable(getattr(self, k))}

    def save(self, filename=None):
        """保存配置快照为json（便于实验复现/论文记录/比对）"""
        save_path = filename or Path(self.RESULT_DIR) / "experiment_config.json"
        Path(self.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def show(self):
        print("="*10 + " 当前实验配置 " + "="*10)
        for k, v in self.to_dict().items():
            print(f"{k:16}: {v}")
        print("="*34)

# 单例出口（直接import config.cfg用）
cfg = Config()