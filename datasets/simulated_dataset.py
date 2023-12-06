import numpy as np
import time
import torch
from torch.utils.data import Dataset

class MatrixDataset(Dataset):
    def __init__(self, target_matrix, length):
        """
        Args:
            target_matrix (numpy.ndarray): A target 3D matrix for operations.
            length (int): Number of samples in the dataset.
            dimensions (tuple): Dimensions (n, m, k) for the randomly generated matrices.
        """
        #self.target_matrix = target_matrix
        self.target_matrix = target_matrix / np.sum(np.abs(target_matrix))  # Normalize the target matrix
        self.length = length
        self.dimensions = target_matrix.shape  # Dimensions for the random matrices
        self.seed = int(time.time())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np.random.seed(self.seed + idx)  # Ensures different random data for each index
        # Generate a random n x n matrix from a uniform distribution between 0 and 1
        random_matrix = np.random.uniform(0, 1, self.dimensions)
        random_matrix = torch.from_numpy(random_matrix).float()
        random_matrix = random_matrix.unsqueeze(0)  # Add channel dimension

        # Compute the dot product with the target matrix
        target_tensor = torch.from_numpy(self.target_matrix).float()
        output_matrix = random_matrix.squeeze(0)*target_tensor
        output_value = output_matrix.sum()

        # Apply sigmoid function to the output_value
        #output_value = torch.tensor(output_value).clone().detach()

        return random_matrix, output_value