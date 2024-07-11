from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import numpy as np
import open_clip
import json
from clip_benchmark.metrics.zeroshot_retrieval import recall_at_k, batchify, dataloader_with_indices
from clip_benchmark.datasets.builder import get_dataset_collate_fn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    model_name: str
    retrieval_json_dir: str
    retrieval_images_dir: str = ""
    remoteclip_path: str = None
    batch_size: int = 64
    workers: int = 8
    device: str = "cuda"


class RetrievalEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.model, self.preprocess, self.tokenize = self.get_model()

    def get_model(self):
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=self.config.model_name,
            pretrained='openai',
            device=self.config.device,
            cache_dir='/home/nw/Codes/RemoteCLIP/cache/weights/open_clip'
        )
        tokenize = open_clip.tokenize
        checkpoint = torch.load(self.config.remoteclip_path, map_location=self.config.device)
        msg = CLIP_model.load_state_dict(checkpoint)
        print(msg)
        return CLIP_model, preprocess_val, tokenize

    def load_dataset(self):
        return JsonDataset(
            self.config.retrieval_json_dir,
            self.config.retrieval_images_dir,
            self.preprocess
        )

    def create_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.workers,
            collate_fn=get_dataset_collate_fn('mscoco_captions')  # return image_captions_collate_fnimage_captions_collate_fn
        )

    def compute_embeddings(self, dataloader):
        batch_images_emb_list = []
        batch_texts_emb_list = []
        texts_image_index = []
        dataloader = dataloader_with_indices(dataloader)

        for batch_images, batch_texts, inds in tqdm(dataloader):
            batch_images = batch_images.to(self.config.device)
            batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
            batch_texts = self.tokenize([text for texts in batch_texts for text in texts]).to(self.config.device)

            with torch.no_grad():
                batch_image_features = self.model.encode_image(batch_images)
                batch_text_features = self.model.encode_text(batch_texts)
                batch_images_emb = F.normalize(batch_image_features, dim=-1)
                batch_texts_emb = F.normalize(batch_text_features, dim=-1)

            batch_images_emb_list.append(batch_images_emb.cpu())
            batch_texts_emb_list.append(batch_texts_emb.cpu())
            texts_image_index.extend(batch_texts_image_index)

        images_emb = torch.cat(batch_images_emb_list)
        texts_emb = torch.cat(batch_texts_emb_list)

        return images_emb, texts_emb, texts_image_index

    def evaluate_retrieval(self, recall_k_list):
        dataset = self.load_dataset()
        dataloader = self.create_dataloader(dataset)
        images_emb, texts_emb, texts_image_index = self.compute_embeddings(dataloader)

        scores = texts_emb @ images_emb.t()
        positive_pairs = torch.zeros_like(scores, dtype=bool)
        positive_pairs[torch.arange(len(scores)), texts_image_index] = True
        metrics = {}

        for recall_k in recall_k_list:
            metrics[f"retrieval-image2text-R@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, self.config.batch_size, self.config.device, k=recall_k) > 0).float().mean().item() * 100

        for recall_k in recall_k_list:
            metrics[f"retrieval-text2image-R@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, self.config.batch_size, self.config.device, k=recall_k) > 0).float().mean().item() * 100

        metrics["retrieval-mean-recall"] = np.mean(list(metrics.values()))

        for key in metrics:
            metrics[key] = round(metrics[key], 2)

        return metrics


class JsonDataset(Dataset):
    def __init__(self, json_dir, img_dir, transforms):
        self.json_dir = json_dir
        self.transforms = transforms
        self.img_dir = img_dir
        self.images = []
        self.captions = []
        self.read_json()
        self.duplicate()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        texts = self.captions[idx]
        return image, texts

    def read_json(self):
        with open(self.json_dir, "r") as f:
            datasets = json.load(f)
        for image in datasets['images']:
            if image['split'] == "test":
                for text in image['sentences']:
                    self.images.append(image['filename'])
                    self.captions.append(text['raw'].capitalize())

    def duplicate(self):
        unique_images, indices = np.unique(self.images, return_index=True)
        if len(unique_images) != len(self.images):
            duplicated_images = []
            duplicated_captions = []
            for index in indices:
                duplicated_images.append(self.images[index])
                same_indices = [i for i, x in enumerate(self.images) if x == self.images[index]]
                captions = [self.captions[same_index] for same_index in same_indices]
                duplicated_captions.append(captions)

            self.images = duplicated_images
            self.captions = duplicated_captions


def main():

    # 定义路径作为常量
    ROOT_DIR = '/home/nw/Codes/RemoteCLIP'
    DATASET_DIR = os.path.join(ROOT_DIR, 'Datasets/RSITMD')
    CHECKPOINTS_DIR = os.path.join(ROOT_DIR, 'checkpoints')

    # 构建Config对象
    config = Config(
        model_name='ViT-B-32',  # 设置为你的模型名
        retrieval_json_dir=os.path.join(DATASET_DIR, 'dataset_RSITMD.json'),  # JSON路径
        retrieval_images_dir=os.path.join(DATASET_DIR, 'images'),  # 图片路径
        remoteclip_path=os.path.join(CHECKPOINTS_DIR, 'RemoteCLIP-ViT-B-32.pt'),  # 模型权重路径
        batch_size=64,
        workers=8
    )

    # 判断设备是否可用
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建评估器
    evaluator = RetrievalEvaluator(config)
    
    # 执行评价
    recall_k_list=[1, 5, 10]
    metrics = evaluator.evaluate_retrieval(recall_k_list)

    # 打印结果
    for name, val in metrics.items():
        print(f"{name}: {val}")


if __name__ == "__main__":
    main()