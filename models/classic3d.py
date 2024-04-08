import torch
import torch.nn as nn
import torch.nn.functional as F


class Classic3d(nn.Module):
    def __init__(self, input_depth, input_height, input_width, conv3d_out_channels=10, conv2_out_channels=64, conv2_1st_layer_kernel=3, conv2_2nd_layer_kernel=3, conv2_3rd_layer_kernel=3):
        super(Classic3d, self).__init__()

        # 3D convolutional layer to handle depth
        self.avgpool3d = nn.AvgPool3d((1, 2, 2))
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=conv3d_out_channels, kernel_size=(input_depth, 1, 1), stride=1, padding=0)
        self.bn3d = nn.BatchNorm2d(num_features=conv3d_out_channels)

        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=conv3d_out_channels, out_channels=conv2_out_channels, kernel_size=conv2_1st_layer_kernel, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=conv2_out_channels)

        self.conv2 = nn.Conv2d(in_channels=conv2_out_channels, out_channels=conv2_out_channels, kernel_size=conv2_2nd_layer_kernel, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_out_channels)

        self.conv3 = nn.Conv2d(in_channels=conv2_out_channels, out_channels=conv2_out_channels, kernel_size=conv2_3rd_layer_kernel, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=conv2_out_channels)

        # 1x1 Convolution to project the channels to a single channel
        self.conv1x1 = nn.Conv2d(in_channels=conv2_out_channels, out_channels=1, kernel_size=1)

        # Use a dummy input to calculate the size for the linear layer
        self._to_linear_size = self._get_linear_input_size(input_depth, input_height, input_width)

        # Linear projection layer
        self.fc = nn.Linear(self._to_linear_size, 1)

    def _get_linear_input_size(self, input_depth, input_height, input_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_depth, input_height, input_width)
            dummy_input = self.avgpool3d(dummy_input)
            dummy_input = self.conv3d(dummy_input).squeeze(2)
            dummy_input = self.bn3d(dummy_input)
            dummy_input = F.relu(self.bn1(self.conv1(dummy_input)))
            dummy_input = F.relu(self.bn2(self.conv2(dummy_input)))
            dummy_input = F.relu(self.bn3(self.conv3(dummy_input)))
            dummy_input = self.conv1x1(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        x = self.avgpool3d(x)
        x = self.conv3d(x).squeeze(2)
        x = self.bn3d(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Apply 1x1 convolution
        x = self.conv1x1(x)

        # Flatten the output for the linear layer
        x = torch.flatten(x, start_dim=1)

        # Linear projection
        x = self.fc(x)
        return x
