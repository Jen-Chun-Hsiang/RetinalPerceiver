import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RetinalCNN(nn.Module):
    def __init__(self, input_depth, input_height, input_width, output_size=1, hidden_size=128,
                 device=None, conv3d_out_channels=10):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.conv3d_out_channels = conv3d_out_channels

        # 3D convolutional layer to handle depth
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=self.conv3d_out_channels,
                                kernel_size=(input_depth, 1, 1), stride=1, padding=0).to(self.device)

        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.conv3d_out_channels, out_channels=32,
                               kernel_size=3, stride=1, padding=1).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)

        # Calculate the size of the flattened output after the last pooling layer
        self._to_linear = self._get_conv_output(input_depth, input_height, input_width)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, output_size).to(self.device)

    def _get_conv_output(self, input_depth, input_height, input_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_depth, input_height, input_width, device=self.device)
            dummy_input = self.conv3d(dummy_input)
            new_height, new_width = dummy_input.size(3), dummy_input.size(4)
            dummy_input = dummy_input.view(1, self.conv3d_out_channels, new_height, new_width)
            dummy_input = self.pool(F.relu(self.conv1(dummy_input)))
            dummy_input = self.pool(F.relu(self.conv2(dummy_input)))
            return int(np.prod(dummy_input.size()[1:]))

    def forward(self, x):
        x = x.to(self.device)

        # 3D convolutional layer
        x = self.conv3d(x)
        new_height, new_width = x.size(3), x.size(4)
        x = x.view(-1, self.conv3d_out_channels, new_height, new_width)

        # 2D convolutional layers with pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(-1, self._to_linear)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
