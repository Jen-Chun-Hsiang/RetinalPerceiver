import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import numpy as np
import logging
import time
import os
from io import StringIO
import sys
from torchinfo import summary

from datasets.simulated_target_rf import MultiTargetMatrixGenerator, CellClassLevel, ExperimentalLevel, IntegratedLevel
from utils.utils import plot_and_save_3d_matrix_with_timestamp as plot3dmat
from utils.utils import SeriesEncoder
from datasets.simulated_dataset import MultiMatrixDataset
from models.perceiver3d import RetinalPerceiverIO
from models.cnn3d import RetinalCNN
from utils.training_procedure import Trainer, Evaluator, save_checkpoint, load_checkpoint

def parse_covariance(string):
    try:
        # Split the string into list of strings
        values = string.split(',')
        # Check if we have exactly four values
        if len(values) != 4:
            raise ValueError
        # Convert each string to float
        values = [float(val) for val in values]
        # Form a 2x2 matrix
        cov_matrix = np.array(values).reshape(2, 2)
        return cov_matrix
    except:
        raise argparse.ArgumentTypeError("Covariance matrix must be four floats separated by commas (e.g., '0.12,0.05,0.04,0.03')")


def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')
    parser.add_argument('--model', type=str, choices=['RetinalPerceiver', 'RetinalCNN'], required=True,
                        help='Model to train')
    parser.add_argument('--input_depth', type=int, default=20, help='Number of time points')
    parser.add_argument('--input_height', type=int, default=30, help='Heights of the input')
    parser.add_argument('--input_width', type=int, default=40, help='Width of the input')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of color channel')
    parser.add_argument('--total_length', type=int, default=1000, help='Number of simulated data')
    parser.add_argument('--train_proportion', type=float, default=0.8, help='Proportion for training data split')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden nodes (information bottleneck)')
    parser.add_argument('--output_size', type=int, default=1, help='Number of neurons for prediction')
    parser.add_argument('--conv3d_out_channels', type=int, default=10, help='Number of temporal in CNN3D')
    # Training procedure
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/model.pth', help='Path to save load model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Flag to load the model from checkpoint')
    # Target matrix specificity
    parser.add_argument('--sf_surround_weight', type=float, default=0.5, help='Strength of spatial surround')
    parser.add_argument('--tf_surround_weight', type=float, default=0.2, help='Strength of temporal surround')
    parser.add_argument("--mean", nargs=2, type=float, default=(0.1, -0.2), help="Mean as two separate floats (e.g., 0.1 -0.2)")
    parser.add_argument("--mean2", nargs=2, type=float, default=None, help="Mean as two separate floats (e.g., 0.1 -0.2)")
    parser.add_argument("--cov", type=parse_covariance, default=np.array([[0.12, 0.05], [0.04, 0.03]]),
                        help="Covariance matrix as four floats separated by commas (e.g., '0.12,0.05,0.04,0.03')")
    parser.add_argument("--cov2", type=parse_covariance, default=None,
                        help="Covariance matrix as four floats separated by commas (e.g., '0.12,0.05,0.04,0.03')")
    # Stimulus specificity
    parser.add_argument('--stimulus_type', type=int, default=4, help='Stimulus type')
    parser.add_argument('--stimulus_type_set', nargs='+', type=int, default=[1], help='Sets of stimulus type')
    # Perceiver specificity
    parser.add_argument('--num_head', type=int, default=4, help='Number of heads in perceiver')
    parser.add_argument('--num_iter', type=int, default=1, help='Number of input reiteration')
    parser.add_argument('--num_latent', type=int, default=16, help='Number of latent length (encoding)')
    parser.add_argument('--num_band', type=int, default=10, help='Number of bands in positional encoding')
    parser.add_argument('--use_layer_norm', action='store_true', help='Enable layer normalization')
    # Plot parameters
    parser.add_argument('--num_cols', type=int, default=5, help='Number of columns in a figure')

    return parser.parse_args()

def main():
    args = parse_args()
    filename_fixed = args.experiment_name
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    # Generate a timestamp
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Construct the full path for the log file
    log_filename = os.path.join(saveprint_dir, f'{filename_fixed}_training_log_{timestr}.txt')

    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")

    # If CUDA is available, continue with the rest of the script
    device = torch.device("cuda")

    # Create cells and cell classes
    cell_class1 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]),
                                 sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                 sf_weight_surround=0.5, num_cells=1, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

    # Create experimental level with cell classes
    experimental = ExperimentalLevel(tf_weight_surround=0.2, tf_sigma_center=0.05,
                                     tf_sigma_surround=0.12, tf_mean_center=0.08,
                                     tf_mean_surround=0.12, tf_weight_center=1,
                                     tf_offset=0, cell_classes=[cell_class1])

    # Create integrated level with experimental levels
    integrated_list = IntegratedLevel([experimental])

    # Generate param_list
    param_list, series_ids = integrated_list.generate_combined_param_list()

    # Encode series_ids into query arrays
    max_values = {'Experiment': 10, 'Type': 10, 'Cell': 10}
    lengths = {'Experiment': 2, 'Type': 2, 'Cell': 2}
    shuffle_components = ['Cell']
    query_encoder = SeriesEncoder(max_values, lengths, shuffle_components=shuffle_components)
    query_array = query_encoder.encode(series_ids)
    logging.info(f'query_array size:{query_array.shape} \n')
    # Use param_list in MultiTargetMatrixGenerator
    multi_target_gen = MultiTargetMatrixGenerator(param_list)
    target_matrix = multi_target_gen.create_3d_target_matrices(
        input_height=args.input_height, input_width=args.input_width, input_depth=args.input_depth)

    logging.info(f'target matrix: cov2:{args.cov2} spatial surround weight:{args.sf_surround_weight} \n')
    # plot and save the target_matrix figure
    plot3dmat(target_matrix[0, :, :, :], args.num_cols, savefig_dir, file_prefix='plot_3D_matrix')
    #plot3dmat(target_matrix[2, :, :, :], args.num_cols, savefig_dir, file_prefix='plot_3D_matrix')

    # Initialize the dataset with the device
    dataset = MultiMatrixDataset(target_matrix, length=args.total_length, device=device,
                                       combination_set=args.stimulus_type_set)

    # Splitting the dataset into training and validation sets
    train_length = int(0.8 * args.total_length)  # 80% for training
    val_length = args.total_length - train_length  # 20% for validation

    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, Loss, and Optimizer
    if args.model == 'RetinalPerceiver':
        model = RetinalPerceiverIO(input_dim = args.input_channels, latent_dim= args.hidden_size, output_dim=args.output_size,
                                   num_latents=args.num_latent, heads=args.num_head,depth=args.num_iter, query_dim=query_array.shape[1],
                                   depth_dim=args.input_depth, height=args.input_height, width=args.input_width,
                                   num_bands=args.num_band, device=device, use_layer_norm=args.use_layer_norm)
    elif args.model == 'RetinalCNN':
        model = RetinalCNN(args.input_depth, args.input_height, args.input_width, args.output_size,
                           hidden_size=args.hidden_size, device=device, conv3d_out_channels=args.conv3d_out_channels)
    logging.info(f'Model: {args.model} \n')
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    model_input = [torch.rand(1, args.input_channels, args.input_depth, args.input_height, args.input_width),
                   torch.rand(1, 1, query_array.shape[1])]
    summary(model, input_data=model_input)
    sys.stdout = old_stdout
    logging.info(buffer.getvalue())

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Initialize the Trainer
    trainer = Trainer(model, criterion, optimizer, device)
    # Initialize the Evaluator
    evaluator = Evaluator(model, criterion, device)

    # Optionally, load from checkpoint
    if args.load_checkpoint:
        start_epoch, model, optimizer, training_losses, validation_losses = load_checkpoint(args.checkpoint_path, model,
                                                                                            optimizer, device)
    else:
        start_epoch = 0
        training_losses = []
        validation_losses = []
        start_time = time.time()  # Capture the start time

    for epoch in range(start_epoch, args.epochs):
        avg_train_loss = trainer.train_one_epoch(train_loader, epoch, query_array)
        training_losses.append(avg_train_loss)

        avg_val_loss = evaluator.evaluate(val_loader, query_array)
        validation_losses.append(avg_val_loss)

        # Print training status
        if (epoch + 1) % 5 == 0:
            elapsed_time = time.time() - start_time
            # Log the epoch and elapsed time, and on a new indented line, log the losses
            logging.info(f"{filename_fixed} Epoch [{epoch + 1}/{args.epochs}], Elapsed time: {elapsed_time:.2f} seconds\n"
                         f"\tTraining Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:  # Example: Save every 10 epochs
            checkpoint_filename = f'{filename_fixed}_checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(epoch, model, optimizer, training_losses, validation_losses, os.path.join(savemodel_dir, checkpoint_filename))

if __name__ == '__main__':
    main()
