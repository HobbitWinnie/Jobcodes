import torch
import open_clip
from PIL import Image
from dataclasses import dataclass
import os

@dataclass
class Config:
    model_name: str
    checkpoint_path: str
    image_dir: str
    text_query: str
    device: str = 'cuda'

class TexttoImageRetrieval:
    def __init__(self, config: Config):
        """Initializes the CLIP model for text-to-image retrieval."""
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create the model and preprocessing transformations.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(config.model_name)
        self.tokenizer = open_clip.get_tokenizer(config.model_name)

        # Load model weights.
        self._load_checkpoint(config.checkpoint_path)

        # Set image directory and text query.
        self.image_dir = config.image_dir
        self.text_query = config.text_query

        # Move model to the specified device and set to evaluation mode.
        self.model = self.model.to(self.device).eval()

    def _load_checkpoint(self, checkpoint_path: str):
        """Loads the model weights from the checkpoint."""
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            message = self.model.load_state_dict(ckpt, strict=False)
            print(f"Loaded model checkpoint with message: {message}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {e}")

    def preprocess_images(self):
        """Loads and preprocesses all images in the directory."""
        image_tensors = []
        image_paths = []
        try:
            for fname in os.listdir(self.image_dir):
                path = os.path.join(self.image_dir, fname)
                if os.path.isfile(path) and path.endswith(('png', 'jpg', 'jpeg')):
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    image_tensors.append(image_tensor)
                    image_paths.append(path)
            return torch.cat(image_tensors), image_paths
        except Exception as e:
            raise RuntimeError(f"Failed to open or preprocess images: {e}")

    def tokenize_text(self) -> torch.Tensor:
        """Tokenizes the text query."""
        try:
            return self.tokenizer([self.text_query]).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize text: {e}")

    def predict(self) -> str:
        """Predicts the most similar image to the text query."""
        try:
            image_tensors, image_paths = self.preprocess_images()
            text_tokens = self.tokenize_text()

            # Enable no_grad and mixed precision inference
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.model.encode_image(image_tensors)
                text_features = self.model.encode_text(text_tokens)

                # Normalize feature vectors
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity
                similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1).cpu().numpy()[0]

            # Get the image with the highest probability
            best_image_idx = similarity.argmax()
            best_image_path = image_paths[best_image_idx]
            best_prob = similarity[best_image_idx]

            return best_image_path, best_prob
        except Exception as e:
            raise RuntimeError(f"Failed to predict: {e}")


# 用法示例
if __name__ == "__main__":
    # Define input parameters
    model_name = 'ViT-L-14'
    checkpoint_path = f"/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-{model_name}.pt"

    config = Config(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        image_dir='/home/nw/Codes/RemoteCLIP/assets/',  # Directory containing images
        text_query="A busy airport with many airplanes.",
        device='cuda'  # or 'cpu' depending on your environment
    )

    # Run the CLIP model for text-to-image retrieval
    retrieval_model = TexttoImageRetrieval(config)
    best_image_path, best_prob = retrieval_model.predict()

    print(f'Best match for the query "{config.text_query}":')
    print(f"Image path: {best_image_path}")
    print(f"Probability: {best_prob * 100:5.1f}%")