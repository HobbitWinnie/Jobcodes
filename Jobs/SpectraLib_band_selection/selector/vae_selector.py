import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def train_vae(X_np, latent_dim=8, epochs=50, batch_size=32, device='cpu'):
    input_dim = X_np.shape[1]
    model = VAE(input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model.train()
    for epo in range(epochs):
        for batch, in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def vae_band_ranking(model, X_np, device='cpu'):
    """依次mask每个波段，比较重建损失变化"""
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(X)
        base_loss = nn.MSELoss()(recon, X).item()
        delta_losses = []
        for d in range(X_np.shape[1]):
            X_mask = X.clone()
            X_mask[:, d] = 0
            recon_m, mu_m, logvar_m = model(X_mask)
            loss = nn.MSELoss()(recon_m, X).item()
            delta_losses.append(loss - base_loss)
        ranked = torch.argsort(torch.tensor(delta_losses), descending=True).cpu().numpy()
    return ranked, delta_losses