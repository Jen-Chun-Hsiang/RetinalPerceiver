import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchRenorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, r_d_max_inc_step=0.001):
        super(BatchRenorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.r_d_max_inc_step = r_d_max_inc_step

        # Initialize parameters
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))  # Add this line
        self.register_buffer('r_max', torch.ones(num_features))
        self.register_buffer('d_max', torch.zeros(num_features))

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    def forward(self, x):
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            batch_mean = x.mean(dim=dims)
            batch_std = x.std(dim=dims, unbiased=True)

            r = (batch_std / self.running_std.view_as(batch_std)).clamp(1 / self.r_max, self.r_max)
            d = ((batch_mean - self.running_mean.view_as(batch_mean)) / self.running_std.view_as(batch_std)).clamp(
                -self.d_max, self.d_max)

            x = (x - batch_mean[None, :, None, None]) / batch_std[None, :, None, None] * \
                r[None, :, None, None] + d[None, :, None, None]

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std

            self.r_max = (self.r_max + self.r_d_max_inc_step * self.num_batches_tracked).clamp(1.0, 3.0)
            self.d_max = (self.d_max + 5 * self.r_d_max_inc_step * self.num_batches_tracked + 0.25).clamp(0.0, 5.0)
            self.num_batches_tracked += 1

        else:
            with torch.no_grad():
                x = (x - self.running_mean[None, :, None, None]) / (self.running_std[None, :, None, None] + self.eps)

        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")


# Example usage:
# batch_renorm = BatchRenorm2d(num_features=64)
# input_tensor = torch.randn(32, 64, 28, 28)  # Batch size of 32, 64 channels, and 28x28 images
# output = batch_renorm(input_tensor)
