import torch
import numpy as np
from torch.utils.data import Dataset
import time

import torch
import numpy as np
from torch.utils.data import Dataset
import time
import random

class MatrixDataset(Dataset):
    def __init__(self, target_matrix, length, device, matrix_type=1, combination_set=None):
        """
        Args:
            target_matrix (numpy.ndarray): A target 3D matrix for operations.
            length (int): Number of samples in the dataset.
            device (torch.device): Device where the tensors will be stored.
            matrix_type (int): Type of random matrix to generate.
            combination_set (list): Set of types to combine for type 4.
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
        self.matrix_type = matrix_type
        self.combination_set = combination_set if combination_set is not None else [1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np.random.seed(self.seed + idx)

        if self.matrix_type == 4:
            selected_type = random.choice(self.combination_set)
            random_matrix = self.generate_matrix(selected_type)
        else:
            random_matrix = self.generate_matrix(self.matrix_type)

        random_matrix = torch.tensor(random_matrix, dtype=torch.float32, device=self.device)
        random_matrix = random_matrix.unsqueeze(0)

        output_matrix = random_matrix.squeeze(0) * self.target_matrix
        output_value = output_matrix.sum()

        return random_matrix, output_value

    def generate_matrix(self, matrix_type):
        if matrix_type == 1:
            return np.random.uniform(0, 1, self.dimensions)
        elif matrix_type == 2:
            return np.random.randint(0, 2, self.dimensions).astype(float)
        elif matrix_type == 3:
            base = np.random.uniform(0, 1, (self.dimensions[0],))
            return np.tile(base, (self.dimensions[1], 1)).T
        else:
            raise ValueError("Invalid matrix type")

'''
# Example usage
dataset = MatrixDataset(target_matrix=np.array([[[1, 2], [3, 4]]]), 
                        length=100, 
                        device=torch.device('cpu'), 
                        matrix_type=4, 
                        combination_set=[1, 2, 3])
'''

