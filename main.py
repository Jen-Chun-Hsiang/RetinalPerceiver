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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/'

    # Example usage
    dims = [10, 20, 30]  # Size of the matrix
    length = 1000  # Number of samples in the dataset
    target_matrix = np.random.randn(*dims)  # Random target matrix
    sample_matrix, output_sum = dataset[0]

    dataset = MatrixDataset(target_matrix, length)
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = f'{savemodel_dir}annmodel_checkpoint_{timestr}.pth'
    torch.save({'dims': dims, 'length': length, 'target_matrix': target_matrix,
                'sample_matrix': sample_matrix, 'output_sum': output_sum,
                }, filename)

    plt.savefig(f'{p.result_dir}check_figure{cfid}.png')