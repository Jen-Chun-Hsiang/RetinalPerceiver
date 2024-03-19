import torch
import torch.nn as nn
import torch.nn.functional as F


class UniqueIdEncoder(nn.Module):
    def __init__(self, num_to_encode, embedding_dim):
        super(UniqueIdEncoder, self).__init__()
        self.feature_embedding = nn.Embedding(num_embeddings=num_to_encode, embedding_dim=embedding_dim)

    def forward(self, unique_ids):
        embeddings = self.feature_embedding(unique_ids)
        return embeddings


class AdaptiveBatchNorm(nn.Module):
    def __init__(self, num_features, embedding_size, eps=1e-5, momentum=0.1):
        super(AdaptiveBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Shared running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Replace gamma and beta with linear layers for embedding projection
        self.fc_gamma = nn.Linear(embedding_size, num_features)
        self.fc_beta = nn.Linear(embedding_size, num_features)

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
        nn.init.normal_(self.fc_gamma.weight, 1.0, 0.02)
        nn.init.zeros_(self.fc_beta.weight)

    def forward(self, x, embeddings):
        if self.training:
            self.num_batches_tracked += 1
            exponential_average_factor = self.momentum if self.momentum is not None else 1.0 / float(
                self.num_batches_tracked)

            # Compute current batch statistics
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)

            # Update running statistics using EMA
            self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * mean
            self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize using either current batch or running statistics
        x_hat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)

        # Generate dataset-specific scale (gamma) and shift (beta) using embeddings
        gamma = self.fc_gamma(embeddings).view(-1, self.num_features, 1, 1)
        beta = self.fc_beta(embeddings).view(-1, self.num_features, 1, 1)

        return gamma * x_hat + beta


# NeuronSpecificSpatialAttention definition from previous context
class NeuronSpecificSpatialAttention(nn.Module):
    def __init__(self, height, width, embedding_size):
        super(NeuronSpecificSpatialAttention, self).__init__()
        self.fc_spatial_gamma = nn.Linear(embedding_size, height * width)
        self.fc_spatial_beta = nn.Linear(embedding_size, height * width)

    def forward(self, x, embeddings):
        batch_size, _, H, W = x.shape
        spatial_gamma = self.fc_spatial_gamma(embeddings).view(batch_size, 1, H, W)  # .expand_as(x)
        spatial_beta = self.fc_spatial_beta(embeddings).view(batch_size, 1, H, W)  # .expand_as(x)
        output = spatial_gamma * x + spatial_beta
        return output, spatial_gamma


# NeuronSpecificFeatureModulation definition from previous context
class NeuronSpecificFeatureModulation(nn.Module):
    def __init__(self, num_channels, embedding_size):
        super(NeuronSpecificFeatureModulation, self).__init__()
        self.fc_feature_gamma = nn.Linear(embedding_size, num_channels)
        self.fc_feature_beta = nn.Linear(embedding_size, num_channels)

    def forward(self, x, embeddings):
        batch_size, _, _, _ = x.shape
        feature_gamma = self.fc_feature_gamma(embeddings).view(batch_size, -1, 1, 1)
        feature_beta = self.fc_feature_beta(embeddings).view(batch_size, -1, 1, 1)
        output = feature_gamma * x + feature_beta
        return output, feature_gamma


class FiLMCNN(nn.Module):
    def __init__(self, input_depth, input_height, input_width,
                 conv3d_out_channels=10, conv2_out_channels=64, conv2_1st_layer_kernel=3,
                 conv2_2nd_layer_kernel=3, conv2_3rd_layer_kernel=3, num_dataset=10, num_neuron=10,
                 dataset_embedding_length=10, neuronid_embedding_length=10, momentum=0.1):
        super().__init__()
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.conv3d_out_channels = conv3d_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.conv2_1st_layer_kernel = conv2_1st_layer_kernel
        self.conv2_2nd_layer_kernel = conv2_2nd_layer_kernel
        self.conv2_3rd_layer_kernel = conv2_3rd_layer_kernel
        self.dataset_embedding_length = dataset_embedding_length
        self.neuronid_embedding_length = neuronid_embedding_length
        self.momentum = momentum

        # 3D convolutional layer to handle depth
        self.avgpool3d = nn.AvgPool3d((1, 2, 2))
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=self.conv3d_out_channels,
                                kernel_size=(input_depth, 1, 1), stride=1, padding=0)
        self.bn3d = AdaptiveBatchNorm(num_features=self.conv2_out_channels,
                                      embedding_size=self.dataset_embedding_length,
                                      momentum=self.momentum)  # Batch normalization for 3D conv

        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.conv3d_out_channels, out_channels=self.conv2_out_channels,
                               kernel_size=self.conv2_1st_layer_kernel, stride=1)
        self.bn1 = AdaptiveBatchNorm(num_features=self.conv2_out_channels, embedding_size=self.dataset_embedding_length,
                                     momentum=self.momentum)  # Batch normalization for 2D conv1
        self.conv2 = nn.Conv2d(in_channels=self.conv2_out_channels, out_channels=self.conv2_out_channels,
                               kernel_size=self.conv2_2nd_layer_kernel, stride=1)
        self.bn2 = AdaptiveBatchNorm(num_features=self.conv2_out_channels, embedding_size=self.dataset_embedding_length,
                                     momentum=self.momentum)
        self.conv3 = nn.Conv2d(in_channels=self.conv2_out_channels, out_channels=self.conv2_out_channels,
                               kernel_size=self.conv2_2nd_layer_kernel, stride=1)
        self.bn3 = AdaptiveBatchNorm(num_features=self.conv2_out_channels, embedding_size=self.dataset_embedding_length,
                                     momentum=self.momentum)

        # Calculate the size of the flattened output after the last pooling layer
        self._to_proj_h, self._to_proj_w = self._get_conv_output(input_depth, input_height, input_width)

        # neuron specific projection layers
        self.feamap = NeuronSpecificFeatureModulation(num_channels=self.conv2_out_channels,
                                                      embedding_size=self.neuronid_embedding_length)
        self.spamap = NeuronSpecificSpatialAttention(height=self._to_proj_h, width=self._to_proj_w,
                                                     embedding_size=self.neuronid_embedding_length)
        self.dataset_id_encoder = UniqueIdEncoder(num_to_encode=num_dataset,
                                                  embedding_dim=self.dataset_embedding_length)
        self.neuron_id_encoder = UniqueIdEncoder(num_to_encode=num_neuron, embedding_dim=self.neuronid_embedding_length)

    def _get_conv_output(self, input_depth, input_height, input_width):
        with torch.no_grad():
            dummy_dataset_embeddings = torch.randn(1, self.dataset_embedding_length)
            dummy_input = torch.zeros(1, 1, input_depth, input_height, input_width)
            dummy_input = self.avgpool3d(dummy_input)
            dummy_input = self.conv3d(dummy_input).squeeze(2)
            dummy_input = self.bn3d(dummy_input, dummy_dataset_embeddings)
            dummy_input = F.softplus(self.bn1(self.conv1(dummy_input), dummy_dataset_embeddings))
            dummy_input = F.softplus(self.bn2(self.conv2(dummy_input), dummy_dataset_embeddings))
            dummy_input = F.softplus(self.bn3(self.conv3(dummy_input), dummy_dataset_embeddings))
            return dummy_input.shape[-2], dummy_input.shape[-1]

    def forward(self, x, dataset_ids, neuron_ids):
        # Get embedding from unique ids
        dataset_embeddings = self.dataset_id_encoder(dataset_ids)
        neuron_embeddings = self.neuron_id_encoder(neuron_ids)

        # 3D convolutional layer
        x = self.avgpool3d(x)
        x = self.conv3d(x).squeeze(2)
        x = self.bn3d(x, dataset_embeddings)

        # 2D convolutional layers with pooling
        x = F.softplus(self.bn1(self.conv1(x), dataset_embeddings))
        x = F.softplus(self.bn2(self.conv2(x), dataset_embeddings))
        x = F.softplus(self.bn3(self.conv3(x), dataset_embeddings))

        # Flatten the output for the fully connected layer
        x, feature_gamma = self.feamap(x, neuron_embeddings)
        x = F.softplus(x).sum(dim=1, keepdim=True)
        x, spatial_gamma = self.spamap(x, neuron_embeddings)
        x = F.softplus(x).sum(dim=(2, 3), keepdim=True)
        return F.softplus(x).sum(dim=(1, 2, 3)), feature_gamma, spatial_gamma
