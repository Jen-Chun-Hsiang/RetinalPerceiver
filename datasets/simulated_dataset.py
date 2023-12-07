import torch
import numpy as np
from torch.utils.data import Dataset
import time

class MatrixDataset(Dataset):
    def __init__(self, target_matrix, length, device):
        """
        Args:
            target_matrix (numpy.ndarray): A target 3D matrix for operations.
            length (int): Number of samples in the dataset.
            device (torch.device): Device where the tensors will be stored.
        """
        # Convert target_matrix to a PyTorch tensor if it's a numpy array
        if isinstance(target_matrix, np.ndarray):
            target_matrix = torch.tensor(target_matrix, dtype=torch.float32, device=device)
        elif isinstance(target_matrix, torch.Tensor):
            target_matrix = target_matrix.to(device)

        # Normalize the target matrix
        norm_factor = torch.sum(torch.abs(target_matrix))
        if norm_factor != 0:
            self.target_matrix = target_matrix / norm_factor
        else:
            self.target_matrix = target_matrix

        self.length = length
        self.dimensions = target_matrix.shape  # Dimensions for the random matrices
        self.device = device
        self.seed = int(time.time())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np.random.seed(self.seed + idx)  # Ensures different random data for each index

        # Generate a random n x n matrix from a uniform distribution between 0 and 1
        random_matrix = np.random.uniform(0, 1, self.dimensions)
        random_matrix = torch.tensor(random_matrix, dtype=torch.float32, device=self.device)
        random_matrix = random_matrix.unsqueeze(0)  # Add channel dimension

        # Compute the dot product with the target matrix
        output_matrix = random_matrix.squeeze(0) * self.target_matrix
        output_value = output_matrix.sum()

        return random_matrix, output_value
