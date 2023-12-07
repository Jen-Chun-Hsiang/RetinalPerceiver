import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
import time
from datasets.simulated_dataset import MatrixDataset
from models.perceiver3d import Perceiver
from torchsummary import summary
import sys

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    # Example usage
    dims = [10, 20, 30]  # Size of the matrix
    length = 1000  # Number of samples in the dataset
    target_matrix = np.random.randn(*dims)  # Random target matrix

    dataset = MatrixDataset(target_matrix, length)
    sample_matrix, output_sum = dataset[0]
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = f'{savemodel_dir}annmodel_checkpoint_{timestr}.pth'
    torch.save({'dims': dims, 'length': length, 'target_matrix': target_matrix,
                'sample_matrix': sample_matrix, 'output_sum': output_sum,
                }, filename)

    # Example to test the model
    input_size = 1  # example input size (i.e., channels)
    output_size = 1  # example output size, e.g., for classification into 10 classes

    model = Perceiver(input_size, 128, output_size, 32, 4, 1, 20, 30, 40, 10)
    model = model.to(device)  # Move model to GPU
    input_tensor = torch.rand(32, 1, 20, 30, 40)  # Example input tensor for n x n image
    output = model(input_tensor)

    # Check the output shape and values
    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to a file
    with open(f'{saveprint_dir}model_summary_{timestr}.txt', 'w') as file:
        sys.stdout = file
        # Print the summary to file
        if model.device == output.device
            summary(model, (1, 20, 30, 40))
        print("Output Shape:", output.shape)  # should be [1, output_size]
        print("Output Values:", output)  # values should be between 0 and 1 due to sigmoid
