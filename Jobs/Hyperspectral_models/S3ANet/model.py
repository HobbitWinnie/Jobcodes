
import torch
import torch.nn as nn
import torch.nn.functional as F


# Spectral Attention Module (SAM)
class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 8)
        self.fc2 = nn.Linear(in_channels // 8, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling on spatial dimensions
        b, c, h, w = x.size()
        avg_pool = x.view(b, c, -1).mean(dim=2)  # [B, C]
        attention = self.fc1(avg_pool)
        attention = F.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        return x * attention  # Element-wise multiplication


# Scale Attention Module (for multi-scale feature aggregation)
class ScaleAttention(nn.Module):
    def __init__(self, in_channels, scales):
        super(ScaleAttention, self).__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=s, dilation=s)
            for s in scales
        ])
        self.attention_weights = nn.Parameter(torch.ones(len(scales)))

    def forward(self, x):
        out = 0
        weights = F.softmax(self.attention_weights, dim=0)  # Normalize scale weights
        for i, layer in enumerate(self.attentions):
            out += weights[i] * layer(x)
        return out


# Spatial Attention Module (SAM)
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)  # Create spatial attention map
        attention = self.sigmoid(attention)  # Normalize to [0, 1]
        return x * attention  # Element-wise multiplication


# Encoder-Decoder Structure for S3ANet
class S3ANet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(S3ANet, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.spectral_attention1 = SpectralAttention(64)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.spectral_attention2 = SpectralAttention(128)

        # Multi-scale module with Scale Attention
        self.scale_attention = ScaleAttention(in_channels=128, scales=[1, 2, 4])

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.spatial_attention1 = SpatialAttention(64)

        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.spatial_attention2 = SpatialAttention(32)

        # Final output layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.encoder1(x)
        x1 = self.spectral_attention1(x1)

        x2 = self.encoder2(x1)
        x2 = self.spectral_attention2(x2)

        # Multi-scale module
        x2 = self.scale_attention(x2)

        # Decoder path
        d1 = self.decoder1(x2)
        d1 = self.spatial_attention1(d1 + x1)  # Skip connection with spatial attention

        d2 = self.decoder2(d1)
        d2 = self.spatial_attention2(d2)

        # Final segmentation/classification
        output = self.final_conv(d2)
        return output


# Additive Angular Margin Loss (AAM Loss)
class AAMLoss(nn.Module):
    def __init__(self, margin=0.5, scale=30):
        super(AAMLoss, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits, labels):
        # Normalize logits
        logits = F.normalize(logits, dim=1)
        # Add margin to the correct class
        one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
        logits_with_margin = logits - one_hot * self.margin
        # Scale logits
        scaled_logits = logits_with_margin * self.scale
        return F.cross_entropy(scaled_logits, labels)


# Example usage
if __name__ == "__main__":
    # Model initialization
    model = S3ANet(in_channels=128, num_classes=5)  # For H2 imagery (e.g., 128 bands)
    input_tensor = torch.randn(1, 128, 256, 256)  # Example input: [B, C, H, W]
    output = model(input_tensor)
    print("Output shape:", output.shape)

    # Example loss computation
    labels = torch.randint(0, 5, (1, 256, 256))  # Random ground truth labels
    criterion = AAMLoss()
    loss = criterion(output.permute(0, 2, 3, 1).reshape(-1, 5), labels.view(-1))
    print("Loss:", loss.item())

