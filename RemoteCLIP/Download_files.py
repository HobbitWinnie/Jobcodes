# from huggingface_hub import hf_hub_download

# try:
#     # 正确地指定 repo_type
#     file_path = hf_hub_download(repo_id="huggingface/Hub", filename="README.md", repo_type="dataset")
#     print(f"File is downloaded to {file_path}.")
# except Exception as e:
#     print(f"验证失败: {e}")


import requests

urls = [
    ("https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-RN50.pt?download=true", "/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-RN50.pt"),
    ("https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt?download=true", "/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-ViT-B-32.pt"),
    ("https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-L-14.pt?download=true", "/home/nw/Codes/RemoteCLIP/checkpoints/RemoteCLIP-ViT-L-14.pt"),
    ("https://huggingface.co/chendelong/RemoteCLIP/resolve/main/.gitattributes?download=true", "/home/nw/Codes/RemoteCLIP/checkpoints/.gitattributes")
]

def download_file(url, dest_path):
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors
    with open(dest_path, 'wb') as f:
        f.write(response.content)
    print(f'{url} is downloaded to {dest_path}.')

# # Ensure the checkpoints directory exists
# import os
# os.makedirs('checkpoints', exist_ok=True)

for url, dest_path in urls:
    download_file(url, dest_path)