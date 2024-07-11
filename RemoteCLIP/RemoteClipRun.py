import torch
import open_clip
from PIL import Image
from dataclasses import dataclass

@dataclass
class RemoteCLIPConfig:
    model_name: str
    checkpoint_path: str
    image_path: str
    text_queries: list
    device: str = 'cuda'


class RemoteCLIPModel:
    def __init__(self, config: RemoteCLIPConfig):
        """Initializes the CLIP model for image-text retrieval.

        Args:
            model_name (str): Name of the model architecture.
            checkpoint_path (str): Path to the checkpoint file.
            device (str): Device to run the model on. Default is 'cuda'.
        """
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create the model and preprocessing transformations.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(config.model_name)
        self.tokenizer = open_clip.get_tokenizer(config.model_name)

        # Load model weights.
        self._load_checkpoint(config.checkpoint_path)

        # Set image and text queries
        self.image_path = config.image_path
        self.text_queries = config.text_queries

        # Move model to the specified device and set to evaluation mode.
        self.model = self.model.to(self.device).eval()

    def _load_checkpoint(self, checkpoint_path: str):
        """Loads the model weights from the checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            message = self.model.load_state_dict(ckpt, strict=False)
            print(f"Loaded model checkpoint with message: {message}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {e}")

    def preprocess_image(self) -> torch.Tensor:
        """Loads and preprocesses the image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        try:
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            raise RuntimeError(f"Failed to open or preprocess image: {e}")

    def tokenize_text(self) -> torch.Tensor:
        """Tokenizes the text queries.
        Args: text_queries (list): List of text queries.
        Returns: torch.Tensor: Tokenized text tensor.
        """
        try:
            return self.tokenizer(self.text_queries).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize text: {e}")
        
    def predict(self) -> dict:
        """Predicts the similarity between image and text queries.

        Args:
            image_path (str): Path to the input image.
            text_queries (list): List of text queries.

        Returns:
            dict: Predictions with text queries as keys and probabilities as values.
        """
        try:
            image_tensor = self.preprocess_image()
            text_tokens = self.tokenize_text()

            # 启用无梯度和混合精度推理
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)

                # 归一化特征向量
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 计算相似度并转换为概率
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

            return {query: prob for query, prob in zip(self.text_queries, text_probs)}
        except Exception as e:
            raise RuntimeError(f"Failed to predict: {e}")

def run_clip_model(config: RemoteCLIPConfig):
    """Runs the CLIP model for image-text retrieval."""
    clip_model = RemoteCLIPModel(config)
    predictions = clip_model.predict()

    print(f'Predictions of {config.model_name}:')
    for query, prob in predictions.items():
        print(f"{query:<40} {prob * 100:5.1f}%")

if __name__ == "__main__":

    # 定义输入参数
    model_name='ViT-L-14'
    checkpoint_path = f"/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-{model_name}.pt"

    config = RemoteCLIPConfig(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        image_path='/home/nw/Codes/RemoteCLIP/assets/airport.jpg',
        text_queries=[
            "A busy airport with many airplanes.", 
            "Satellite view of Hohai University.", 
            "A building next to a lake.", 
            "Many people in a stadium.", 
            "a cute cat"
            ],
        device='cuda'  # 或者 'cpu' 根据你的设备环境
    )

    # 直接调用函数进行调试
    run_clip_model(config)

