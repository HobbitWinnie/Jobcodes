import torch
import open_clip
from PIL import Image
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str
    checkpoint_path: str
    image_path: str
    text_queries: list
    device: str = 'cuda'

class ImagetoTextRetrieval:
    def __init__(self, config: Config):
        """Initializes the CLIP model for image-text retrieval."""
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
        """Loads the model weights from the checkpoint."""
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            message = self.model.load_state_dict(ckpt, strict=False)
            print(f"Loaded model checkpoint with message: {message}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {e}")

    def preprocess_image(self) -> torch.Tensor:
        """Loads and preprocesses the image."""
        try:
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            raise RuntimeError(f"Failed to open or preprocess image: {e}")

    def tokenize_text(self) -> torch.Tensor:
        """Tokenizes the text queries."""
        try:
            return self.tokenizer(self.text_queries).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize text: {e}")
        
    def predict(self) -> dict:
        """Predicts the similarity between image and text queries."""
        try:
            image_tensor = self.preprocess_image()
            text_tokens = self.tokenize_text()

            # Enable no_grad and mixed precision inference
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)

                # Normalize feature vectors
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity and convert to probabilities
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

            return {query: prob for query, prob in zip(self.text_queries, text_probs)}
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
        image_path='/home/nw/Codes/RemoteCLIP/assets/airport.jpg',
        text_queries=[
            "A busy airport with many airplanes.", 
            "Satellite view of Hohai University.", 
            "A building next to a lake.", 
            "Many people in a stadium.", 
            "a cute cat"
        ],
        device='cuda'  # or 'cpu' depending on your environment
    )

    # Run the CLIP model for image-text retrieval
    retrieval_model = ImagetoTextRetrieval(config)
    predictions = retrieval_model.predict()

    # Find the text query with the highest probability
    best_query = max(predictions, key=predictions.get)
    best_prob = predictions[best_query]

    print(f'Best prediction for {config.model_name}:')
    print(f"{best_query:<40} {best_prob * 100:5.1f}%")