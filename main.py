import torch
import numpy as np
import time
from datasets.simulated_dataset import MatrixDataset
from models.perceiver3d import Perceiver
from torchsummary import summary
import sys

if __name__ == '__main__':
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'

    # Set the device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example usage
    dims = [10, 20, 30]  # Size of the matrix
    length = 1000  # Number of samples in the dataset
    target_matrix = np.random.randn(*dims)  # Random target matrix

    # Initialize the dataset with the device
    dataset = MatrixDataset(target_matrix, length, device)
    sample_matrix, output_sum = dataset[0]

    # Save the dataset information
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = f'{savemodel_dir}annmodel_checkpoint_{timestr}.pth'
    torch.save({'dims': dims, 'length': length, 'target_matrix': target_matrix,
                'sample_matrix': sample_matrix.cpu().numpy(), 'output_sum': output_sum.item(),
                }, filename)

    # Initialize the model
    input_size = 1  # Example input size (i.e., channels)
    output_size = 1  # Example output size, e.g., for classification into 10 classes
    model = Perceiver(input_size, 128, output_size, 32, 4, 1, 20, 30, 40, 10, device=device)

    # Create an input tensor and move it to the device
    input_tensor = torch.rand(32, 1, 20, 30, 40).to(device)  # Example input tensor for n x n image
    output = model(input_tensor)

    # Check the output shape and values
    original_stdout = sys.stdout  # Save the original stdout
    with open(f'{saveprint_dir}model_summary_{timestr}.txt', 'w') as file:
        sys.stdout = file
        # Print the summary to file
        summary(model, (1, 20, 30, 40))
        print("Output Shape:", output.shape)  # Should be [1, output_size]
        print("Output Values:", output)  # Values should be between 0 and 1 due to sigmoid

    sys.stdout = original_stdout  # Restore the original stdout
    print("Model summary has been saved to 'model_summary_{timestr}.txt'.")
