import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_datasets, eps=1e-5, momentum=0.1):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Shared running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Dataset-specific learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_datasets, num_features))
        self.beta = nn.Parameter(torch.zeros(num_datasets, num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x, dataset_ids):
        # Verify dataset_ids are within range
        if torch.any(dataset_ids < 0) or torch.any(dataset_ids >= self.gamma.size(0)):
            print(dataset_ids)
            raise ValueError("dataset_id out of range")

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

        # Apply dataset-specific scale (gamma) and shift (beta)
        gamma = self.gamma[dataset_ids, :].view(-1, self.num_features, 1, 1)
        beta = self.beta[dataset_ids, :].view(-1, self.num_features, 1, 1)
        return gamma * x_hat + beta


class NeuronSpecificSpatialSelection(nn.Module):
    def __init__(self, num_neurons, height, width):
        super(NeuronSpecificSpatialSelection, self).__init__()
        self.height = height
        self.width = width
        # Embedding for each neuron ID, creating a spatial mask of size [height * width]
        self.spatial_embedding = nn.Embedding(num_neurons, height * width)

    def forward(self, x, neuron_ids):
        batch_size, num_channels, _, _ = x.shape

        # Retrieve the spatial masks for the given neuron IDs and reshape
        spatial_masks = self.spatial_embedding(neuron_ids)  # Shape: [batch_size, height * width]
        spatial_masks = spatial_masks.view(batch_size, 1, self.height, self.width)  # Add channel dim for broadcasting

        # Use torch.einsum to apply the spatial mask and aggregate in one step
        # 'bchw,bchw->bc' notation means: for each batch (b) and channel (c),
        # multiply elements of height (h) and width (w) and then sum over h and w.
        output = torch.einsum('bchw,bchw->bc', x, spatial_masks)

        return output


class NeuronSpecificFeatureSelection(nn.Module):
    def __init__(self, num_neurons, num_channels):
        super(NeuronSpecificFeatureSelection, self).__init__()
        # Embedding for each neuron ID, creating a spatial mask of size num_channels
        self.channel_embedding = nn.Embedding(num_neurons, num_channels)

    def forward(self, x, neuron_ids):
        batch_size, num_channels, _, _ = x.shape

        # Retrieve the spatial masks for the given neuron IDs and reshape
        channel_masks = self.channel_embedding(neuron_ids)  # Shape: [batch_size, num_channels]
        channel_masks = channel_masks.view(batch_size, num_channels, 1, 1)  # Add height and width dim for broadcasting

        # Use torch.einsum to apply the spatial mask and aggregate in one step
        output = torch.einsum('bchw,bchw->bhw', x, channel_masks).unsqueeze(1)

        return output


class StyleCNN(nn.Module):
    def __init__(self, input_depth, input_height, input_width,
                 conv3d_out_channels=10, conv2_out_channels=64, conv2_1st_layer_kernel=3,
                 conv2_2nd_layer_kernel=3, conv2_3rd_layer_kernel=3,
                 num_dataset=100, momentum=0.1, num_neuron=1000):
        super().__init__()
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.conv3d_out_channels = conv3d_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.conv2_1st_layer_kernel = conv2_1st_layer_kernel
        self.conv2_2nd_layer_kernel = conv2_2nd_layer_kernel
        self.conv2_3rd_layer_kernel = conv2_3rd_layer_kernel
        self.num_dataset = num_dataset
        self.momentum = momentum
        self.num_neuron = num_neuron

        # 3D convolutional layer to handle depth
        self.avgpool3d = nn.AvgPool3d((1, 2, 2))
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=self.conv3d_out_channels,
                                kernel_size=(input_depth, 1, 1), stride=1, padding=0)
        self.bn3d = AdaptiveBatchNorm2d(self.conv3d_out_channels, self.num_dataset, momentum=self.momentum)  # Batch normalization for 3D conv

        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.conv3d_out_channels, out_channels=self.conv2_out_channels,
                               kernel_size=self.conv2_1st_layer_kernel, stride=1)
        self.bn1 = AdaptiveBatchNorm2d(self.conv2_out_channels, self.num_dataset, momentum=self.momentum)  # Batch normalization for 2D conv1
        self.conv2 = nn.Conv2d(in_channels=self.conv2_out_channels, out_channels=self.conv2_out_channels,
                               kernel_size=self.conv2_2nd_layer_kernel, stride=1)
        self.bn2 = AdaptiveBatchNorm2d(self.conv2_out_channels, self.num_dataset, momentum=self.momentum)
        self.conv3 = nn.Conv2d(in_channels=self.conv2_out_channels, out_channels=self.conv2_out_channels,
                               kernel_size=self.conv2_2nd_layer_kernel, stride=1)
        self.bn3 = AdaptiveBatchNorm2d(self.conv2_out_channels, self.num_dataset, momentum=self.momentum)

        # Calculate the size of the flattened output after the last pooling layer
        self._to_proj_h, self._to_proj_w = self._get_conv_output(input_depth, input_height, input_width)

        # neuron specific projection layers
        self.feamap = NeuronSpecificFeatureSelection(num_neurons=self.num_neuron,
                                                     num_channels=self.conv2_out_channels)
        self.spamap = NeuronSpecificSpatialSelection(num_neurons=self.num_neuron,
                                                     height=self._to_proj_h, width=self._to_proj_w)

    def _get_conv_output(self, input_depth, input_height, input_width):
        with torch.no_grad():
            dummy_dataset_ids = torch.randint(0, 100, (1,))
            dummy_input = torch.zeros(1, 1, input_depth, input_height, input_width)
            dummy_input = self.avgpool3d(dummy_input)
            dummy_input = self.conv3d(dummy_input).squeeze(2)
            dummy_input = self.bn3d(dummy_input, dummy_dataset_ids)
            dummy_input = F.softplus(self.bn1(self.conv1(dummy_input), dummy_dataset_ids))
            dummy_input = F.softplus(self.bn2(self.conv2(dummy_input), dummy_dataset_ids))
            dummy_input = F.softplus(self.bn3(self.conv3(dummy_input), dummy_dataset_ids))
            return dummy_input.shape[-2], dummy_input.shape[-1]

    def forward(self, x, dataset_id, neuron_id):
        # 3D convolutional layer
        x = self.avgpool3d(x)
        x = self.conv3d(x).squeeze(2)
        x = self.bn3d(x, dataset_id)

        # 2D convolutional layers with pooling
        x = F.softplus(self.bn1(self.conv1(x), dataset_id))
        x = F.softplus(self.bn2(self.conv2(x), dataset_id))
        x = F.softplus(self.bn3(self.conv3(x), dataset_id))

        # Flatten the output for the fully connected layer
        x = self.feamap(x, neuron_id)
        x = self.spamap(x, neuron_id)
        return F.softplus(x)
