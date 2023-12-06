import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, input_dim, latent_dim, heads):
        super().__init__()
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(input_dim, latent_dim)
        self.value_proj = nn.Linear(input_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, heads)

    def forward(self, x, context):
        q = self.query_proj(x).transpose(0, 1)
        k = self.key_proj(context).transpose(0, 1)
        v = self.value_proj(context).transpose(0, 1)
        attn_output, _ = self.attention(q, k, v)
        attn_output = attn_output.transpose(0, 1)
        return self.out_proj(attn_output)

class SelfAttention(nn.Module):
    def __init__(self, latent_dim, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(latent_dim, heads)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x_transposed = x.transpose(0, 1)
        attn_output, _ = self.attention(x_transposed, x_transposed, x_transposed)
        attn_output = attn_output.transpose(0, 1)
        return attn_output + x
        #return self.norm(attn_output + x)

class FourierFeaturePositionalEncoding3D(nn.Module):
    def __init__(self, depth, height, width, num_bands):
        super().__init__()
        self.num_bands = num_bands

        # Create a grid of 3D coordinates
        z_coord, y_coord, x_coord = torch.meshgrid(
            torch.linspace(-1, 1, depth),
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width)
        )
        coord = torch.stack([z_coord, y_coord, x_coord], dim=0)  # Shape: [3, depth, height, width]

        # Apply Fourier Feature Mapping
        frequencies = 2.0 ** torch.linspace(0., num_bands - 1, num_bands)
        self.frequencies = frequencies.reshape(-1, 1, 1, 1, 1)
        self.fourier_basis = torch.cat([torch.sin(coord * self.frequencies), torch.cos(coord * self.frequencies)], dim=0)

    def forward(self, x):
        a, b, c, d, e = self.fourier_basis.shape
        basis = self.fourier_basis.reshape(1, a*b, c, d, e)
        fourier_features = basis.repeat(x.shape[0], 1, 1, 1, 1)
        return torch.cat([x, fourier_features], dim=1)

class Perceiver(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, num_latents, heads, depth=2, depth_dim=10, height=20, width=20, num_bands=10):
        super().__init__()
        # Calculate the total channels after adding Fourier features
        total_channels = input_dim + num_bands * 2 * 3  # 2 for sin & cos, 3 for depth, height & width

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attentions = nn.ModuleList([CrossAttention(total_channels, latent_dim, heads) for _ in range(depth)])
        self.self_attentions = nn.ModuleList([SelfAttention(latent_dim, heads) for _ in range(depth)])
        self.fc = nn.Linear(latent_dim, output_dim)
        self.positional_encoding = FourierFeaturePositionalEncoding3D(depth_dim, height, width, num_bands)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # Flatten the 3D input
        latents = self.latents.unsqueeze(0).repeat(x.shape[0], 1, 1)

        for cross_attention, self_attention in zip(self.cross_attentions, self.self_attentions):
            latents = cross_attention(latents, x) + latents
            latents = self_attention(latents) + latents

        return self.fc(latents.mean(dim=1))