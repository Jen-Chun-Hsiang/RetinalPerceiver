import torch
import numpy as np
from torch.utils.data import Dataset
import time

import torch
import numpy as np
from torch.utils.data import Dataset
import time
import random

import torch
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class MatrixDataset(Dataset):
    def __init__(self, target_matrix, length, device, combination_set=None, ratio_for_one=0.5):
        """
        Args:
            target_matrix (numpy.ndarray or torch.Tensor): A target 3D matrix for operations.
            length (int): Number of samples in the dataset.
            device (torch.device): Device where the tensors will be stored.
            combination_set (list): Set of matrix types to use for generating random matrices.
            ratio_for_one (float): Ratio for generating binary matrices (Type 2).
        """

        # Convert target_matrix to a PyTorch tensor
        if isinstance(target_matrix, np.ndarray):
            target_matrix = torch.tensor(target_matrix, dtype=torch.float32, device=device)
        elif isinstance(target_matrix, torch.Tensor):
            target_matrix = target_matrix.to(device)

        # Normalize the target matrix
        norm_factor = torch.sum(torch.abs(target_matrix))
        self.target_matrix = target_matrix / norm_factor if norm_factor != 0 else target_matrix

        self.length = length
        self.dimensions = target_matrix.shape
        self.device = device
        self.seed = int(time.time())
        self.ratio_for_one = ratio_for_one
        self.combination_set = combination_set if combination_set is not None else [1]
        self.median_filter = MedianPool2d(kernel_size=4, same=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)

        # Select a matrix type from the combination set
        selected_type = random.choice(self.combination_set)
        random_matrix = self.generate_matrix(selected_type)

        random_matrix = random_matrix.unsqueeze(0)
        output_matrix = random_matrix.squeeze(0) * self.target_matrix
        output_value = output_matrix.sum()

        return random_matrix, output_value

    def generate_matrix(self, matrix_type):
        # Definitions for matrix types 1, 2, and 3 as before
        if matrix_type == 1:
            return torch.rand(self.dimensions, device=self.device)
        elif matrix_type == 2:
            p = self.ratio_for_one
            if not 0 <= p <= 1:
                raise ValueError("Threshold probability p must be between 0 and 1")
            return torch.bernoulli(p * torch.ones(self.dimensions, device=self.device))
        elif matrix_type == 3:
            base = torch.rand(self.dimensions[0], device=self.device)
            replicated_base = base.unsqueeze(1).unsqueeze(2).expand(*self.dimensions)
            return replicated_base
        elif matrix_type == 4:
            base = torch.rand(1, 1, self.dimensions[1], self.dimensions[2], device=self.device)
            base = self.median_filter(base)
            # Corrected expansion along the second dimension
            replicated_base = base.repeat(1, self.dimensions[0], 1, 1)
            return replicated_base.squeeze(0)  # Removing the first dimension to get [depth, height, width]
        else:
            raise ValueError("Invalid matrix type")

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class SharpenFilter(nn.Module):
    def __init__(self):
        super(SharpenFilter, self).__init__()
        # Define a sharpening kernel
        self.sharpening_kernel = torch.tensor([[0, -1, 0],
                                               [-1, 5, -1],
                                               [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # Apply padding
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')  # Reflect padding
        # Apply the sharpening kernel
        x = F.conv2d(x, self.sharpening_kernel, padding=0)  # No additional padding during convolution
        return x

class MultiMatrixDataset(MatrixDataset):
    def __init__(self, target_matrices, length, device, combination_set=None, ratio_for_one=0.5):
        """
        Args:
            target_matrices (list of numpy.ndarray or torch.Tensor): A list of target 3D matrices.
            length (int): Number of samples in the dataset.
            device (torch.device): Device where the tensors will be stored.
            combination_set (list): Set of matrix types to use for generating random matrices.
            ratio_for_one (float): Ratio for generating binary matrices (Type 2).
        """

        # Check if only a single matrix is provided and convert to a list
        if isinstance(target_matrices, (np.ndarray, torch.Tensor)):
            target_matrices = [target_matrices]

        # Initialize the base MatrixDataset with the first matrix (as a placeholder)
        super().__init__(target_matrices[0], length, device, combination_set, ratio_for_one)

        # Normalize and store all matrices
        self.target_matrices = []
        for matrix in target_matrices:
            if isinstance(matrix, np.ndarray):
                matrix = torch.tensor(matrix, dtype=torch.float32, device=device)
            elif isinstance(matrix, torch.Tensor):
                matrix = matrix.to(device)
            else:
                raise ValueError("Target matrices must be either numpy arrays or torch tensors")

            norm_factor = torch.sum(torch.abs(matrix))
            normalized_matrix = matrix / norm_factor if norm_factor != 0 else matrix
            self.target_matrices.append(normalized_matrix)

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)

        # Randomly select a target matrix and its index
        matrix_index = random.randint(0, len(self.target_matrices) - 1)
        selected_matrix = self.target_matrices[matrix_index]
        self.target_matrix = selected_matrix  # Update the target matrix in the base class

        selected_type = random.choice(self.combination_set)
        random_matrix = self.generate_matrix(selected_type, selected_matrix.shape)

        random_matrix = random_matrix.unsqueeze(0)
        output_matrix = random_matrix.squeeze(0) * selected_matrix
        output_value = output_matrix.sum()

        return random_matrix, output_value, matrix_index

