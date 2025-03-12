import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.perceiver3d import PerceiverIODecoder
from utils.utils import add_gradient
from utils.time_manager import TimeFunctionRun


class RetinalCNN(nn.Module):
    def __init__(self, input_depth, input_height, input_width, output_size=1, hidden_size=128, conv3d_out_channels=10,
                 conv2_out_channels=64, conv2_1st_layer_kernel=4, conv2_2nd_layer_kernel=5):
        super().__init__()
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.conv3d_out_channels = conv3d_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.conv2_1st_layer_kernel = conv2_1st_layer_kernel
        self.conv2_2nd_layer_kernel = conv2_2nd_layer_kernel

        # 3D convolutional layer to handle depth
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=self.conv3d_out_channels,
                                kernel_size=(input_depth, 1, 1), stride=1, padding=0)
        self.bn3d = nn.BatchNorm3d(self.conv3d_out_channels)  # Batch normalization for 3D conv

        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.conv3d_out_channels, out_channels=32,
                               kernel_size=self.conv2_1st_layer_kernel, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for 2D conv1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=self.conv2_out_channels,
                               kernel_size=self.conv2_2nd_layer_kernel, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.conv2_out_channels)  # Batch normalization for 2D conv2
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size of the flattened output after the last pooling layer
        self._to_linear = self._get_conv_output(input_depth, input_height, input_width)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def _get_conv_output(self, input_depth, input_height, input_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_depth, input_height, input_width)
            dummy_input = self.conv3d(dummy_input)
            dummy_input = self.bn3d(dummy_input)
            new_height, new_width = dummy_input.size(3), dummy_input.size(4)
            dummy_input = dummy_input.view(1, self.conv3d_out_channels, new_height, new_width)
            dummy_input = F.softplus(self.bn1(self.conv1(dummy_input)))
            dummy_input = self.pool(dummy_input)
            dummy_input = F.softplus(self.bn2(self.conv2(dummy_input)))
            dummy_input = self.pool(dummy_input)
            return int(np.prod(dummy_input.size()[1:]))

    def forward(self, x):
        # 3D convolutional layer
        x = self.conv3d(x)
        x = self.bn3d(x)
        new_height, new_width = x.size(3), x.size(4)
        x = x.view(-1, self.conv3d_out_channels, new_height, new_width)

        # 2D convolutional layers with pooling
        x = F.softplus(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.softplus(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Flatten the output for the fully connected layer
        x = x.view(-1, self._to_linear)

        # Fully connected layers
        x = F.softplus(self.fc1(x))
        x = self.fc2(x)
        return x


class FourierFeaturePositionalEncoding2D(nn.Module):
    def __init__(self, height, width, num_bands):
        super().__init__()
        self.height = height
        self.width = width
        self.num_bands = num_bands

        # Precompute the positional encoding and register it as a buffer
        encoding = self.compute_positional_encoding(height, width, num_bands)
        self.register_buffer('positional_encoding', encoding)

    def get_fourier_features(self, dimension_size):
        coord = torch.linspace(-1, 1, dimension_size)
        coord = coord.unsqueeze(0)  # Shape: [1, dimension_size]

        frequencies = 2.0 ** torch.linspace(0., self.num_bands - 1, self.num_bands)
        frequencies = frequencies.reshape(-1, 1)  # Shape: [num_bands, 1]
        fourier_basis = torch.cat([torch.sin(coord * frequencies), torch.cos(coord * frequencies)], dim=0)

        return fourier_basis

    def compute_positional_encoding(self, height, width, num_bands):
        spatial_features_height = self.get_fourier_features(height)  # Shape: [2*num_bands, height]
        spatial_features_width = self.get_fourier_features(width)  # Shape: [2*num_bands, width]

        spatial_features_height = spatial_features_height.unsqueeze(-1).repeat(1, 1,
                                                                               width)  # Shape: [2*num_bands, height, width]
        spatial_features_width = spatial_features_width.unsqueeze(-2).repeat(1, height,
                                                                             1)  # Shape: [2*num_bands, height, width]

        spatial_features = torch.cat([spatial_features_height, spatial_features_width], dim=0)
        # Shape: [4*num_bands, height, width]

        return spatial_features

    def forward(self):
        # spatial_features_height = self.get_fourier_features(self.height)  # Shape: [2*num_bands, height]
        # spatial_features_width = self.get_fourier_features(self.width)  # Shape: [2*num_bands, width]
        #
        # spatial_features_height = spatial_features_height.unsqueeze(-1).repeat(1, 1, self.width)
        # spatial_features_width = spatial_features_width.unsqueeze(-2).repeat(1, self.height, 1)
        #
        # spatial_features = torch.cat([spatial_features_height, spatial_features_width], dim=0)
        # # Shape: [4*num_bands, height, width]

        return self.positional_encoding


class FrontEndRetinalCNN(RetinalCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # 3D convolutional layer
        x = self.conv3d(x)
        x = self.bn3d(x)
        new_height, new_width = x.size(3), x.size(4)
        x = x.view(-1, self.conv3d_out_channels, new_height, new_width)

        # 2D convolutional layers with pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Stop here to output the feature map instead of flattening and passing through fully connected layers
        return x

    def get_output_dimensions(self, input_depth, input_height, input_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_depth, input_height, input_width)
            dummy_input = self.conv3d(dummy_input)
            dummy_input = self.bn3d(dummy_input)
            new_height, new_width = dummy_input.size(3), dummy_input.size(4)
            dummy_input = dummy_input.view(1, self.conv3d_out_channels, new_height, new_width)
            dummy_input = F.relu(self.bn1(self.conv1(dummy_input)))
            dummy_input = self.pool(dummy_input)
            dummy_input = F.relu(self.bn2(self.conv2(dummy_input)))
            dummy_input = self.pool(dummy_input)
            return dummy_input.size(2), dummy_input.size(3)  # return height and width


class RetinalPerceiverIOWithCNN(nn.Module):
    def __init__(self, input_depth, input_height, input_width, latent_dim=128, output_dim=1, query_dim=6,
                 num_latents=16, heads=4, use_layer_norm=False, num_bands=10, conv3d_out_channels=10,
                 conv2_out_channels=64, conv2_1st_layer_kernel=4, conv2_2nd_layer_kernel=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.query_dim = query_dim
        self.num_latents = num_latents
        self.heads = heads
        self.use_layer_norm = use_layer_norm

        # Initialize the FrontEndRetinalCNN
        self.front_end_cnn = FrontEndRetinalCNN(input_depth=input_depth,
                                                input_height=input_height,
                                                input_width=input_width,
                                                conv3d_out_channels=conv3d_out_channels,
                                                conv2_out_channels=conv2_out_channels,
                                                conv2_1st_layer_kernel=conv2_1st_layer_kernel,
                                                conv2_2nd_layer_kernel=conv2_2nd_layer_kernel)

        # Get the output dimensions of FrontEndRetinalCNN
        cnn_output_height, cnn_output_width = self.front_end_cnn.get_output_dimensions(input_depth, input_height,
                                                                                       input_width)
        # Initialize Fourier Feature Positional Encoding with the correct dimensions
        self.positional_encoding = FourierFeaturePositionalEncoding2D(cnn_output_height, cnn_output_width, num_bands)

        # Calculate the total number of channels after concatenating CNN output and positional encoding
        self.total_channels = conv2_out_channels + 4 * num_bands  # 4*num_bands for sin and cos for both height and width

        # Initialize the linear projection layer
        self.linear_to_num_latent = nn.Linear(cnn_output_height * cnn_output_width, num_latents)

        self.linear_to_latent_dim = nn.Linear(self.total_channels, self.latent_dim)

        # Initialize the Perceiver IO Decoder
        self.decoder = PerceiverIODecoder(latent_dim=self.latent_dim,
                                          query_dim=self.query_dim,
                                          output_dim=output_dim,
                                          heads=self.heads,
                                          use_layer_norm=self.use_layer_norm)

        # Initialize latent array
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        # cheap linear decoder
        self.fc = nn.Linear(num_latents, output_dim)

    # @TimeFunctionRun
    def forward(self, input_array, query_array):
        query_array = query_array
        query_array = query_array.repeat(1, self.num_latents, 1)
        query_array = add_gradient(query_array, dim=1, start=-1, end=1)
        cnn_output = self.front_end_cnn(input_array)
        # Apply positional encoding
        pos_encoding = self.positional_encoding().unsqueeze(0).repeat(cnn_output.size(0), 1, 1, 1)
        # Concatenate the CNN output with the positional encoding
        # print(f"cnn_output device: {cnn_output.device}")
        # print(f"pos_encoding device: {pos_encoding.device}")
        cnn_output_with_pos = torch.cat([cnn_output, pos_encoding], dim=1)

        # Reshape and project the spatial dimensions to num_latents
        batch_size, num_channels, height, width = cnn_output_with_pos.shape
        cnn_output_flattened = cnn_output_with_pos.view(batch_size, num_channels, -1)
        latents_projected = self.linear_to_num_latent(cnn_output_flattened).permute(0, 2, 1)

        # Project the channels to latent_dim
        latents_projected = latents_projected.view(batch_size, self.num_latents, -1)
        latents_projected = self.linear_to_latent_dim(latents_projected)

        predictions, embeddings = self.decoder(latents_projected, query_array)
        return F.softplus(self.fc(predictions.flatten(start_dim=1))), embeddings
        # return F.relu(predictions.mean(dim=1)), embeddings


class qNAPmask(RetinalPerceiverIOWithCNN):
    def __init__(self, input_depth, input_height, input_width, num_masking, masking_pos, masking_type="continuous",
                 **kwargs):
        super().__init__(input_depth, input_height, input_width, **kwargs)
        self.masking_pos = torch.tensor(masking_pos)
        self.masking_type = masking_type
        self.query_embeddings = nn.Embedding(num_masking, len(masking_pos))

    def forward(self, input_array, query_array, masking_ids):
        # Generate embeddings for the masking_ids
        embeddings = self.query_embeddings(masking_ids)

        # Create a mask for valid masking_ids
        valid_mask = (masking_ids != -1).unsqueeze(-1)

        # Replace values in query_array at masking_pos with embeddings
        query_array[:, self.masking_pos] = torch.where(
            valid_mask,
            embeddings,
            query_array[:, self.masking_pos]
        )

        # Pass the modified query_array through the parent forward method
        return super().forward(input_array, query_array)


class QueryEmbeddingCNN(nn.Module):
    def __init__(self, input_depth, input_height, input_width, latent_dim=128, output_dim=1, query_dim=6,
                 num_latents=16, heads=4, use_layer_norm=False, num_bands=10, conv3d_out_channels=10,
                 conv2_out_channels=64, conv2_1st_layer_kernel=4, conv2_2nd_layer_kernel=5, device=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.query_dim = query_dim
        self.num_latents = num_latents
        self.heads = heads
        self.use_layer_norm = use_layer_norm
        self.device = device

        # Initialize the FrontEndRetinalCNN
        self.front_end_cnn = FrontEndRetinalCNN(input_depth=input_depth,
                                                input_height=input_height,
                                                input_width=input_width,
                                                conv3d_out_channels=conv3d_out_channels,
                                                conv2_out_channels=conv2_out_channels,
                                                conv2_1st_layer_kernel=conv2_1st_layer_kernel,
                                                conv2_2nd_layer_kernel=conv2_2nd_layer_kernel)

        # Get the output dimensions of FrontEndRetinalCNN
        cnn_output_height, cnn_output_width = self.front_end_cnn.get_output_dimensions(input_depth, input_height,
                                                                                       input_width)

        # Initialize Fourier Feature Positional Encoding with the correct dimensions
        self.positional_encoding = FourierFeaturePositionalEncoding2D(cnn_output_height, cnn_output_width, num_bands)

        # Calculate the total number of channels after concatenating CNN output and positional encoding
        self.total_channels = conv2_out_channels + 4 * num_bands  # 4*num_bands for sin and cos for both height and width

        # Initialize the linear projection layer
        self.linear_to_num_latent = nn.Linear(cnn_output_height * cnn_output_width, num_latents)

        self.linear_to_latent_dim = nn.Linear(self.total_channels, self.latent_dim)

        # Initialize the Perceiver IO Decoder
        self.decoder = PerceiverIODecoder(latent_dim=self.latent_dim,
                                          query_dim=self.query_dim,
                                          output_dim=output_dim,
                                          heads=self.heads,
                                          use_layer_norm=self.use_layer_norm)

        # Initialize latent array
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        # cheap linear decoder
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, input_array, query_array):
        # Pass input through the Front End CNN
        query_array = query_array
        query_array = query_array.repeat(1, self.num_latents, 1)
        query_array = add_gradient(query_array, dim=1, start=-1, end=1)

        cnn_output = self.front_end_cnn(input_array)
        # Apply positional encoding
        pos_encoding = self.positional_encoding().unsqueeze(0).repeat(cnn_output.size(0), 1, 1, 1).to(self.device)
        # Concatenate the CNN output with the positional encoding
        try:
            cnn_output_with_pos = torch.cat([cnn_output, pos_encoding], dim=1)
        except Exception as e:
            print(e)
            print(cnn_output.device)
            print(pos_encoding.device)

        # Reshape and project the spatial dimensions to num_latents
        batch_size, num_channels, height, width = cnn_output_with_pos.shape
        cnn_output_flattened = cnn_output_with_pos.view(batch_size, num_channels, -1)
        latents_projected = self.linear_to_num_latent(cnn_output_flattened).permute(0, 2, 1)

        # Project the channels to latent_dim
        latents_projected = latents_projected.view(batch_size, self.num_latents, -1)
        latents_projected = self.linear_to_latent_dim(latents_projected)

        # cheap way to skip decoder and make sure everything above is fine
        #return self.fc(latents_projected.mean(dim=1))
        # Decode stage

        predictions, embeddings = self.decoder(latents_projected, query_array)
        '''
        print(f'predictions size: {predictions.shape}') #  torch.Size([32, 256, 1])
        print(f'embeddings size: {embeddings.shape}') # torch.Size([32, 256, 64])
        raise RuntimeError("Script stopped after saving outputs.")
        '''
        return F.softplus(predictions.mean(dim=1)), embeddings


