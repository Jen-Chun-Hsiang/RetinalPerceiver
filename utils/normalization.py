import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchRenorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, r_d_max_inc_step=0.01):
        super(BatchRenorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.r_d_max_inc_step = r_d_max_inc_step

        # Initialize parameters
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('r_max', torch.ones(num_features))
        self.register_buffer('d_max', torch.zeros(num_features))

    def forward(self, input):
        if self.training:
            with torch.no_grad():
                batch_mean = input.mean(dim=[0, 2, 3])
                batch_std = input.std(dim=[0, 2, 3], unbiased=False)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std
                self.num_batches_tracked += 1

                if self.num_batches_tracked > 1:
                    d_new = torch.max(torch.abs(batch_mean - self.running_mean),
                                      torch.abs(batch_std - self.running_std))
                    self.d_max = torch.clamp_max(self.d_max, d_new)
                    self.r_max = torch.clamp_max(self.r_max, self.r_d_max_inc_step * self.num_batches_tracked.item())

        # Normalize input
        normalized_input = (input - self.running_mean[None, :, None, None]) / (
                    self.running_std[None, :, None, None] + self.eps)

        # Apply renormalization
        normalized_input = normalized_input * self.r_max[None, :, None, None] + self.d_max[None, :, None, None]

        return normalized_input


# Example usage:
# batch_renorm = BatchRenorm2d(num_features=64)
# input_tensor = torch.randn(32, 64, 28, 28)  # Batch size of 32, 64 channels, and 28x28 images
# output = batch_renorm(input_tensor)
