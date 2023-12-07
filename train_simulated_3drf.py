import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

# Importing modules from your project
from datasets.simulated_target_rf import TargetMatrixGenerator
from utils.utils import plot_and_save_3d_matrix_with_timestamp as plot3dmat
from datasets.simulated_dataset import MatrixDataset
from models.perceiver3d import Perceiver
#from utils.trainer import train_one_epoch, evaluate
#from utils.utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")

    parser.add_argument('--input_depth', type=int, default=20, help='Number of time points')
    parser.add_argument('--input_height', type=int, default=30, help='Heights of the input')
    parser.add_argument('--input_width', type=int, default=40, help='Width of the input')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of color channel')
    parser.add_argument('--total_length', type=int, default=1000, help='Number of simulated data')
    parser.add_argument('--train_proportion', type=float, default=0.8, help='Proportion for training data split')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden nodes (information bottleneck)')
    parser.add_argument('--output_size', type=int, default=1, help='Number of neurons for prediction')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/model.pth', help='Path to save the model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Flag to load the model from checkpoint')

    parser.add_argument('--num_head', type=int, default=4, help='Number of heads in perceiver')
    parser.add_argument('--num_iter', type=int, default=1, help='Number of input reiteration')
    parser.add_argument('--num_latent', type=int, default=16, help='Number of latent length (encoding)')
    parser.add_argument('--num_band', type=int, default=10, help='Number of bands in positional encoding')

    parser.add_argument('--num_cols', type=int, default=5, help='Number of columns in a figure')


    return parser.parse_args()

def main():
    args = parse_args()
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")

    # If CUDA is available, continue with the rest of the script
    device = torch.device("cuda")

    # Create the target matrix
    generator = TargetMatrixGenerator(mean=(0.1, -0.2), cov=np.array([[0.12, 0.05], [0.04, 0.03]]), device=device)

    # Generate the target matrix
    target_matrix = generator.create_3d_target_matrix(args.input_height, args.input_width, args.input_depth)

    # plot and save the target_matrix figure
    plot3dmat(target_matrix, args.num_cols, savefig_dir, file_prefix='plot_3D_matrix')
    # Initialize the dataset with the device
    dataset = MatrixDataset(target_matrix, args.total_length, device)

    # Splitting the dataset into training and validation sets
    train_length = int(0.8 * args.total_length)  # 80% for training
    val_length = args.total_length - train_length  # 20% for validation

    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, Loss, and Optimizer
    model = Perceiver(args.input_channels, args.hidden_size, args.output_size, args.num_latent, args.num_head,
                      args.num_iter, args.input_depth, args.input_height, args.input_width, args.num_band,
                      device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Optionally, load from checkpoint
    if args.load_checkpoint:
        start_epoch, model, optimizer = load_checkpoint(args.checkpoint_path, model, optimizer, device)
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device)
        evaluate(test_loader, model, device)

        # Save checkpointt
        save_checkpoint(epoch, model, optimizer, args.checkpoint_path)



'''
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = YourDataset(root='./data', train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, loss and optimizer
    model = YourModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Optionally, load from checkpoint
    if args.load_checkpoint:
        start_epoch, model, optimizer = load_checkpoint(args.checkpoint_path, model, optimizer, device)
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device)
        evaluate(test_loader, model, device)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, args.checkpoint_path)
'''
if __name__ == '__main__':
    main()
