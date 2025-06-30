import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SpectralTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64, nhead=4, depth=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        # 输入 shape: (batch, band_dim)
        x = self.input_proj(x).unsqueeze(1)  # (batch, seq=1, embed_dim)
        # 如果你想每个波段看作一个token，可以修改为：(batch, seq=input_dim, 1)形式进入transformer
        x = self.transformer_encoder(x)  # (batch, seq=1, embed_dim)
        x = x.squeeze(1)
        x = self.output_proj(x)
        return x

def train_transformer(X_np, embed_dim=64, epochs=50, batch_size=32, device='cpu'):
    input_dim = X_np.shape[1]
    model = SpectralTransformerEncoder(input_dim, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model.train()
    for e in range(epochs):
        for batch, in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def transformer_band_ranking(model, X_np, device='cpu'):
    """依次mask每个波段，比较重建损失变化"""
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        base_recon = model(X)
        base_loss = nn.MSELoss()(base_recon, X).item()
        delta_losses = []
        for d in range(X_np.shape[1]):
            X_mask = X.clone()
            X_mask[:, d] = 0
            recon_m = model(X_mask)
            loss = nn.MSELoss()(recon_m, X).item()
            delta_losses.append(loss - base_loss)
        ranked = torch.argsort(torch.tensor(delta_losses), descending=True).cpu().numpy()
    return ranked, delta_losses