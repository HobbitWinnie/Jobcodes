import torch, open_clip
from PIL import Image

model_name = 'ViT-L-14' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

ckpt = torch.load(f"/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-{model_name}.pt", map_location="cpu")
message = model.load_state_dict(ckpt)
print(message)

model = model.cuda().eval()

text_queries = [
    "A busy airport with many airplanes.", 
    "Satellite view of Hohai University.", 
    "A building next to a lake.", 
    "Many people in a stadium.", 
    "a cute cat",
    ]
text = tokenizer(text_queries)
image = preprocess(Image.open("/home/nw/Codes/RemoteCLIP/assets/airport.jpg")).unsqueeze(0)
image = preprocess(Image.open("/home/nw/Assets/data/airport.jpg")).unsqueeze(0)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image.cuda())
    text_features = model.encode_text(text.cuda())
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

print(f'Predictions of {model_name}:')
for query, prob in zip(text_queries, text_probs):
    print(f"{query:<40} {prob * 100:5.1f}%")