import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from utils.utils import add_gradient, VideoPatcher
import numpy as np


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
    def __init__(self, num_frames, height, width, num_bands, seed=35, device=None, use_phase_shift=True,
                 use_dense_frequency=False):
        super().__init__()
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_bands = num_bands
        self.seed = seed
        self.device = device if device is not None else torch.device("cpu")
        self.use_phase_shift = use_phase_shift
        self.use_dense_frequency = use_dense_frequency

        # Generate and shuffle phase shifts
        self.phase_shifts = np.linspace(0, torch.pi, self.num_bands)

    def calculate_dense_frequencies(self, num_bands, dimension_size, device):
        max_freq = dimension_size ** 2
        power_constant = np.log(max_freq) / (num_bands - 1)
        return torch.exp(torch.linspace(0, num_bands - 1, num_bands, device=device) * power_constant)

    def get_fourier_features(self, dimension_size):
        if self.use_dense_frequency:
            frequencies = self.calculate_dense_frequencies(self.num_bands, dimension_size, self.device)
        else:
            frequencies = 2.0 ** torch.linspace(0., self.num_bands - 1, self.num_bands, device=self.device)
        frequencies = frequencies.reshape(-1, 1)  # Shape: [num_bands, 1]

        # Create a grid of 1D coordinates
        coord = torch.linspace(-1, 1, dimension_size, device=self.device)
        coord = coord.unsqueeze(0)  # Shape: [1, dimension_size]

        # Phase shifts linearly spaced over 180 degrees (π radians)
        if self.use_phase_shift:
            np.random.shuffle(self.phase_shifts)
            phase_shift = torch.tensor(self.phase_shifts, device=self.device, dtype=torch.float32).reshape(-1, 1)
        else:
            phase_shift = torch.zeros_like(frequencies)

        # Apply Fourier Feature Mapping
        fourier_basis = torch.cat(
            [torch.sin(coord * frequencies + phase_shift), torch.cos(coord * frequencies + phase_shift)],
            dim=0)

        # Normalize fourier_basis to [-1, 1]
        fourier_basis = 2 * (fourier_basis - fourier_basis.min()) / (fourier_basis.max() - fourier_basis.min()) - 1

        return fourier_basis

    def forward(self):
        np.random.seed(self.seed)
        temporal_features = self.get_fourier_features(self.num_frames)  # Shape: [2*num_bands, num_frames]
        spatial_features_height = self.get_fourier_features(self.height)  # Shape: [2*num_bands, height]
        spatial_features_width = self.get_fourier_features(self.width)  # Shape: [2*num_bands, width]

        return temporal_features, spatial_features_height, spatial_features_width


class FourierFeaturePositionalEncoding3Dadd(nn.Module):
    def __init__(self, num_frames, height, width, num_bands, seed=35, device=None):
        super().__init__()
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_bands = num_bands
        self.seed = seed
        self.device = device if device is not None else torch.device("cpu")

        frequencies = 2.0 ** torch.linspace(0., self.num_bands - 1, self.num_bands, device=self.device)
        self.frequencies = frequencies.reshape(-1, 1)  # Shape: [num_bands, 1]

        # Generate and shuffle phase shifts
        self.phase_shifts = np.linspace(0, torch.pi, self.num_bands)

    def get_fourier_features(self, dimension_size):
        coord = torch.linspace(-1, 1, dimension_size, device=self.device)
        coord = coord.unsqueeze(0)  # Shape: [1, dimension_size]

        phase_shifts = self.phase_shifts
        np.random.shuffle(self.phase_shifts)
        # Phase shifts linearly spaced over 180 degrees (π radians)
        phase_shift = torch.tensor(phase_shifts, device=self.device, dtype=torch.float32).reshape(-1, 1)
        fourier_basis = torch.cat([torch.sin(coord * self.frequencies + phase_shift), torch.cos(coord * self.frequencies + phase_shift)], dim=0)

        return fourier_basis.sum(0)

    def forward(self):
        np.random.seed(self.seed)
        temporal_features = self.get_fourier_features(self.num_frames)
        spatial_features_height = self.get_fourier_features(self.height)
        spatial_features_width = self.get_fourier_features(self.width)

        # Using broadcasting to create a 3D grid
        combined_features = temporal_features[:, None, None] + spatial_features_height[None, :, None] + spatial_features_width[None, None, :]
        return combined_features

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
            #q = self.norm_q(q)
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
        # F.gelu(self.out_proj2(attn_output)), attn_output

        print(f'attn_output size: {attn_output.shape}')  # torch.Size([32, 256, 1])

        return self.out_proj2(attn_output), attn_output


class RetinalPerceiverIO(nn.Module):
    def __init__(self, input_dim=1, latent_dim=128, output_dim=1, depth_dim=20, height=30, width=40,
                 query_dim=6, num_latents=16, heads=4, depth=1, num_bands=10, kernel_size=(2, 2, 2),
                 stride=(1, 1, 1), device=None, concatenate_positional_encoding=False, use_layer_norm=False,
                 use_phase_shift=False, use_dense_frequency=False):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concatenate_positional_encoding = concatenate_positional_encoding
        self.use_phase_shift = use_phase_shift
        self.use_dense_frequency = use_dense_frequency

        # Initialize VideoPatcher
        self.video_patcher = VideoPatcher(video_shape=(1, input_dim, depth_dim, height, width),
                                          kernel_size=kernel_size, stride=stride)
        self.patch_dim = torch.prod(torch.tensor(kernel_size)) * input_dim
        self.patch_indices = self.video_patcher.generate_patch_indices()
        fourier_feature_size = num_bands * 2 * 3  # 2 for sin & cos, 3 for frames, height and width

        # Calculate the size of the Fourier features and Positional Encoding
        positional_encoder_class = FourierFeaturePositionalEncoding3Dindep
        self.positional_encoder = positional_encoder_class(
            num_frames=self.video_patcher.t_patches, height=self.video_patcher.h_patches,
            width=self.video_patcher.w_patches, num_bands=num_bands, device=self.device,
            use_phase_shift=self.use_phase_shift, use_dense_frequency=self.use_dense_frequency
        )

        self.total_channels = self.patch_dim + (fourier_feature_size if concatenate_positional_encoding else 0)
        if not concatenate_positional_encoding:
            self.position_encoding_proj = nn.Linear(fourier_feature_size, self.patch_dim).to(self.device)

        self._initialize_layers(latent_dim, output_dim, query_dim, num_latents, heads, depth, use_layer_norm)

    def _initialize_layers(self, latent_dim, output_dim, query_dim, num_latents, heads, depth, use_layer_norm):
        """ Helper function to initialize layers """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.query_dim = query_dim
        self.num_latents = num_latents
        self.heads = heads
        self.depth = depth
        self.use_layer_norm = use_layer_norm

        # The encoder, process, and decoder parts
        self.encoder = PerceiverIOEncoder(self.total_channels, self.latent_dim, self.heads, self.use_layer_norm).to(
            self.device)
        self.process_layers = nn.ModuleList(
            [PerceiverIOSelfAttention(self.latent_dim, self.heads, self.use_layer_norm).to(self.device) for _ in
                range(self.depth)])
        self.decoder = PerceiverIODecoder(latent_dim=self.latent_dim, query_dim=self.query_dim,
                                          output_dim=self.output_dim, heads=self.heads,
                                          use_layer_norm=self.use_layer_norm).to(self.device)

        # Initialize latent array
        self.latents = nn.Parameter(torch.randn(num_latents, self.latent_dim)).to(self.device)

    def forward(self, input_array, query_array):
        query_array = query_array.to(self.device).repeat(1, self.num_latents, 1)
        query_array = add_gradient(query_array, 1)
        input_array = input_array.to(self.device)

        # Convert input array to patches
        patches = self.video_patcher.video_to_patches(input_array)
        num_batches, C, num_patches, patch_values = patches.shape

        # Process positional encoding
        encoded_input = self._process_positional_encoding(num_batches, num_patches, patches)

        # Encode, Process, and Decode stages
        latents = self.latents.unsqueeze(0).repeat(num_batches, 1, 1)
        latents, _ = self.encoder(encoded_input, latents)

        for layer in self.process_layers:
            latents, _ = layer(latents)

        outputs, _ = self.decoder(latents, query_array)
        return outputs.mean(dim=1)

    def _process_positional_encoding(self, num_batches, num_patches, patches):
        """ Helper function to process positional encoding """
        temporal_encoding, spatial_encoding_height, spatial_encoding_width = self.positional_encoder()
        encoded_input = patches.view(num_batches, num_patches, -1)
        if self.concatenate_positional_encoding:
            encoded_input = self._concatenate_positional_encoding(encoded_input, num_batches, temporal_encoding,
                                                                  self.patch_indices)
        else:
            positional_encoding = self._add_positional_encoding(num_batches, num_patches, temporal_encoding,
                                                                self.patch_indices)
            encoded_input += positional_encoding
        return encoded_input

    def _concatenate_positional_encoding(self, encoded_input, num_batches, temporal_encoding, patch_indices):
        """ Helper function to concatenate positional encoding """
        for i in range(3):
            a = temporal_encoding[:, patch_indices[:, 0]]
            a = rearrange(a, 'a b -> 1 b a').repeat(num_batches, 1, 1)
            encoded_input = torch.cat([encoded_input, a], dim=2)
        return encoded_input

    def _add_positional_encoding(self, num_batches, num_patches, temporal_encoding, patch_indices):
        """ Helper function to add positional encoding """
        positional_encoding = torch.empty(num_batches, num_patches, 0).to(self.device)
        for i in range(3):
            a = temporal_encoding[:, patch_indices[:, 0]]
            a = rearrange(a, 'a b -> 1 b a').repeat(num_batches, 1, 1)
            positional_encoding = torch.cat([positional_encoding, a], dim=2)

        positional_encoding = self.position_encoding_proj(positional_encoding)

        return positional_encoding