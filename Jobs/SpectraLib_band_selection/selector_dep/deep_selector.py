# deep_selector.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def train_autoencoder(X_np, embed_dim=8, epochs=50, batch_size=16, device='cpu'):
    input_dim = X_np.shape[1]
    model = SimpleAutoEncoder(input_dim, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X_torch = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_torch), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch, in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def autoencoder_band_ranking(model, X_np, device='cpu'):
    """每个波段置零（mask），看重建损失增量，用于排序“重要波段”"""
    X_torch = torch.tensor(X_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        base_loss = nn.MSELoss()(model(X_torch)[0], X_torch).item()
        delta_losses = []
        for d in range(X_np.shape[1]):
            X_mask = X_torch.clone()
            X_mask[:, d] = 0
            recon, _ = model(X_mask)
            loss = nn.MSELoss()(recon, X_torch).item()
            delta_losses.append(loss - base_loss)
        ranked = np.argsort(-np.array(delta_losses))  # delta大的优先
    return ranked, delta_losses