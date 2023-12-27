import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, input_dim, latent_dim, heads, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(input_dim, latent_dim)
        self.value_proj = nn.Linear(input_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, heads)
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, context):
        q = self.query_proj(x)
        k = self.key_proj(context)
        v = self.value_proj(context)

        if self.use_layer_norm:
            q = self.norm(q)
            k = self.norm(k)
            v = self.norm(v)

        q, k, v = map(lambda t: t.transpose(0, 1), (q, k, v))
        attn_output, _ = self.attention(q, k, v)
        attn_output = attn_output.transpose(0, 1)
        return self.out_proj(attn_output)

class SelfAttention(nn.Module):
    def __init__(self, latent_dim, heads, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.attention = nn.MultiheadAttention(latent_dim, heads)
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x_transposed = x.transpose(0, 1)

        if self.use_layer_norm:
            x_transposed = self.norm(x_transposed)

        attn_output, _ = self.attention(x_transposed, x_transposed, x_transposed)
        attn_output = attn_output.transpose(0, 1)
        return attn_output + x

class FourierFeaturePositionalEncoding3D(nn.Module):
    def __init__(self, depth, height, width, num_bands, device=None):
        super().__init__()
        self.num_bands = num_bands
        self.device = device if device is not None else torch.device("cpu")

        # Create a grid of 3D coordinates
        z_coord, y_coord, x_coord = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=self.device),
            torch.linspace(-1, 1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device)
        )
        coord = torch.stack([z_coord, y_coord, x_coord], dim=0)  # Shape: [3, depth, height, width]

        # Apply Fourier Feature Mapping
        frequencies = 2.0 ** torch.linspace(0., num_bands - 1, num_bands, device=self.device)
        self.frequencies = frequencies.reshape(-1, 1, 1, 1, 1)
        self.fourier_basis = torch.cat([torch.sin(coord * self.frequencies), torch.cos(coord * self.frequencies)], dim=0)

    def forward(self, x):
        x = x.to(self.device)
        a, b, c, d, e = self.fourier_basis.shape
        basis = self.fourier_basis.reshape(1, a*b, c, d, e)
        fourier_features = basis.repeat(x.shape[0], 1, 1, 1, 1)
        return torch.cat([x, fourier_features], dim=1)


class FourierFeaturePositionalEncoding3Dindep(nn.Module):
    def __init__(self, num_frames, height, width, num_bands, device=None):
        super().__init__()
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_bands = num_bands
        self.device = device if device is not None else torch.device("cpu")

    def get_fourier_features(self, dimension_size):
        # Create a grid of 1D coordinates
        coord = torch.linspace(-1, 1, dimension_size, device=self.device)
        coord = coord.unsqueeze(0)  # Shape: [1, dimension_size]

        # Apply Fourier Feature Mapping
        frequencies = 2.0 ** torch.linspace(0., self.num_bands - 1, self.num_bands, device=self.device)
        frequencies = frequencies.reshape(-1, 1)  # Shape: [num_bands, 1]
        fourier_basis = torch.cat([torch.sin(coord * frequencies), torch.cos(coord * frequencies)], dim=0)

        return fourier_basis

    def forward(self):
        temporal_features = self.get_fourier_features(self.num_frames)  # Shape: [2*num_bands, num_frames]
        spatial_features_height = self.get_fourier_features(self.height)  # Shape: [2*num_bands, height]
        spatial_features_width = self.get_fourier_features(self.width)  # Shape: [2*num_bands, width]

        return temporal_features, spatial_features_height, spatial_features_width


class RetinalPerceiver(nn.Module):
    def __init__(self, input_dim=1, latent_dim=128, output_dim=1, num_latents=16, heads=4, depth=1,
                 depth_dim=20, height=30, width=40, num_bands=10, device=None, use_layer_norm=False):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_channels = input_dim + num_bands * 2 * 3  # 2 for sin & cos, 3 for depth, height & width

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim)).to(self.device)
        self.cross_attentions = nn.ModuleList([CrossAttention(total_channels, latent_dim, heads,
                                                              use_layer_norm=use_layer_norm).to(self.device) for _ in range(depth)])
        self.self_attentions = nn.ModuleList([SelfAttention(latent_dim, heads,
                                                            use_layer_norm=use_layer_norm).to(self.device) for _ in range(depth)])
        self.fc = nn.Linear(latent_dim, output_dim).to(self.device)
        self.positional_encoding = FourierFeaturePositionalEncoding3D(depth_dim, height, width, num_bands, self.device)
        #self.fc_hidden = nn.Linear(latent_dim, latent_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.positional_encoding(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # Flatten the 3D input
        latents = self.latents.unsqueeze(0).repeat(x.shape[0], 1, 1)

        for cross_attention, self_attention in zip(self.cross_attentions, self.self_attentions):
            latents = cross_attention(latents, x) + latents
            #latents = F.relu(self.fc_hidden(latents)) + latents
            latents = self_attention(latents) + latents

        return self.fc(latents.mean(dim=1))


class PerceiverIOEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, heads, use_layer_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.heads = heads
        self.use_layer_norm = use_layer_norm

        # Initialize projections for query, key, value
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(input_dim, latent_dim)
        self.value_proj = nn.Linear(input_dim, latent_dim)

        # Cross-attention
        self.attention = nn.MultiheadAttention(latent_dim, heads)

        # Optional layer normalization
        if self.use_layer_norm:
            self.norm_q = nn.LayerNorm(latent_dim)
            self.norm_k = nn.LayerNorm(latent_dim)
            self.norm_v = nn.LayerNorm(latent_dim)

    def forward(self, input_array, latent_array):
        # Project the query, key, and value
        q = self.query_proj(latent_array)
        k = self.key_proj(input_array)
        v = self.value_proj(input_array)

        # Apply optional normalization
        if self.use_layer_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
            v = self.norm_v(v)

        # Transpose for the attention module: from (batch, seq, feature) to (seq, batch, feature)
        q, k, v = map(lambda t: rearrange(t, 'b n d -> n b d'), (q, k, v))

        # Apply the attention
        updated_latent_array, attention_scores = self.attention(q, k, v)

        # Revert transpose operation
        updated_latent_array = rearrange(updated_latent_array, 'n b d -> b n d')

        # Add the residual connection
        return updated_latent_array + latent_array, attention_scores


class PerceiverIOSelfAttention(nn.Module):
    def __init__(self, latent_dim, heads, use_layer_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.heads = heads
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(latent_dim)

        # Self-attention
        self.attention = nn.MultiheadAttention(latent_dim, heads)

        # Optional layer normalization
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(latent_dim)

    def forward(self, latent_array):
        # If layer normalization is used, apply it before the attention
        if self.use_layer_norm:
            latent_array = self.norm(latent_array)

        # Transpose for the attention module: from (batch, seq, feature) to (seq, batch, feature)
        latent_array_transposed = rearrange(latent_array, 'b n d -> n b d')

        # Apply the self-attention
        updated_latent_array, attention_scores = self.attention(latent_array_transposed, latent_array_transposed,
                                                                latent_array_transposed)

        # Revert transpose operation
        updated_latent_array = rearrange(updated_latent_array, 'n b d -> b n d')

        # Return the updated latent array and the attention scores
        return latent_array+updated_latent_array, attention_scores


class PerceiverIODecoder(nn.Module):
    def __init__(self, latent_dim, query_dim, output_dim, heads, use_layer_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.query_dim = query_dim
        self.output_dim = output_dim
        self.heads = heads
        self.use_layer_norm = use_layer_norm

        # Initialize projections for query, key, value
        self.query_proj = nn.Linear(query_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj1 = nn.Linear(latent_dim, latent_dim)
        self.out_proj2 = nn.Linear(latent_dim, output_dim)

        # Cross-attention
        self.attention = nn.MultiheadAttention(latent_dim, heads)

        # Optional layer normalization
        if self.use_layer_norm:
            self.norm_q = nn.LayerNorm(latent_dim)
            self.norm_k = nn.LayerNorm(latent_dim)
            self.norm_v = nn.LayerNorm(latent_dim)

    def forward(self, latents, query):
        # Project the query, key, and value
        q = self.query_proj(query)
        k = self.key_proj(latents)
        v = self.value_proj(latents)
        saved_q = q.clone()

        # Apply optional normalization
        if self.use_layer_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
            v = self.norm_v(v)

        # Transpose for the attention module: from (batch, seq, feature) to (seq, batch, feature)
        q, k, v = map(lambda t: rearrange(t, 'b n d -> n b d'), (q, k, v))

        # Apply the attention
        attn_output, attn_output_weights = self.attention(q, k, v)

        # Revert transpose operation
        attn_output = rearrange(attn_output, 'n b d -> b n d')
        #attn_output = attn_output + saved_q
        # Apply the output projection
        attn_output = F.gelu(self.out_proj1(attn_output))
        return self.out_proj2(attn_output)


class RetinalPerceiverIO(nn.Module):
    def __init__(self, input_dim=1, latent_dim=128, output_dim=1, depth_dim=20, height=30, width=40,
                 query_dim=6, num_latents=16, heads=4, depth=1, num_bands=10, device=None, use_layer_norm=False):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize positional encoding
        self.positional_encoding = FourierFeaturePositionalEncoding3Dindep(depth_dim, height, width, num_bands, self.device)

        # Calculate the size of the Fourier features for each dimension
        fourier_feature_size = (1 + height + width) * num_bands * 2  # 2 for sin & cos
        self.total_channels = input_dim * height * width + fourier_feature_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.query_dim = query_dim
        self.num_latents = num_latents
        self.heads = heads
        self.depth = depth
        self.use_layer_norm = use_layer_norm

        # The encoder part
        self.encoder = PerceiverIOEncoder(self.total_channels, self.latent_dim, self.heads, self.use_layer_norm).to(
            self.device)

        # The process (self-attention) part, applied multiple times
        self.process_layers = nn.ModuleList([
            PerceiverIOSelfAttention(self.latent_dim, self.heads, self.use_layer_norm).to(self.device) for _ in
            range(self.depth)
        ])

        # The decoder part
        self.decoder = PerceiverIODecoder(latent_dim=self.latent_dim, query_dim=self.query_dim,
                                          output_dim=self.output_dim, heads=self.heads,
                                          use_layer_norm=self.use_layer_norm).to(self.device)

        # Initialize latent array
        self.latents = nn.Parameter(torch.randn(num_latents, self.latent_dim)).to(self.device)
        # cheap linear decoder
        self.fc = nn.Linear(latent_dim, output_dim).to(self.device)
    def forward(self, input_array, query_array):
        query_array = query_array.to(self.device).repeat(1, self.num_latents, 1)
        input_array = input_array.to(self.device)

        # Apply positional encoding to the input
        temporal_encoding, spatial_encoding_height, spatial_encoding_width = self.positional_encoding()

        B, C, T, H, W = input_array.shape
        input_array = rearrange(input_array, 'b c t h w -> b t (c h w)')
        # Adjust spatial encoding: repeat for each frame in each batch
        spatial_encoding = torch.cat([spatial_encoding_height, spatial_encoding_width], dim=1)
        spatial_encoding = spatial_encoding.unsqueeze(0).unsqueeze(0)  # [1, 1, height + width, num_bands * 2]
        spatial_encoding = spatial_encoding.repeat(B, T, 1, 1)  # [B, T, height + width, num_bands * 2]
        spatial_encoding = spatial_encoding.view(B, T, -1)  # [B, T, (height + width) * num_bands * 2]
        # Adjust temporal encoding: repeat for each batch
        temporal_encoding = temporal_encoding.unsqueeze(0).repeat(B, 1, 1)  # [B, num_frames, num_bands * 2]
        temporal_encoding = rearrange(temporal_encoding, 'b c t -> b t c')
        # Concatenate with input
        encoded_input = torch.cat([input_array, spatial_encoding, temporal_encoding], dim=2)
        # Encode stage
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)

        latents, _ = self.encoder(encoded_input, latents)

        # Process stage
        for layer in self.process_layers:
            latents, _ = layer(latents)

        # cheap way to skip decoder and make sure everything above is fine
        #return self.fc(latents.mean(dim=1))
        # Decode stage
        return self.decoder(latents, query_array)