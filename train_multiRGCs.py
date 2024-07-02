import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import numpy as np
import logging
import time
from io import StringIO
import sys
import pandas as pd
from scipy.io import savemat
# from torchinfo import summary
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel

from datasets.neuron_dataset import RetinalDataset, DataConstructor, BatchGenerator
from datasets.neuron_dataset import train_val_split, load_data_from_excel, filter_and_merge_data
from utils.utils import plot_and_save_3d_matrix_with_timestamp as plot3dmat
from utils.utils import SeriesEncoder
from models.perceiver3d import RetinalPerceiverIO
from models.cnn3d import RetinalPerceiverIOWithCNN
from utils.training_procedure import Trainer, Evaluator, save_checkpoint, CheckpointLoader
from utils.loss_function import loss_functions

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
        raise argparse.ArgumentTypeError(
            "Covariance matrix must be four floats separated by commas (e.g., '0.12,0.05,0.04,0.03')")


def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--config_name', type=str, default='neuro_exp1_2cell_030624', help='Config file name for data generation')
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')
    parser.add_argument('--model', type=str, choices=['RetinalPerceiver', 'RetinalCNN'], required=True,
                        help='Model to train')
    parser.add_argument('--input_depth', type=int, default=20, help='Number of time points')
    parser.add_argument('--input_height', type=int, default=30, help='Heights of the input')
    parser.add_argument('--input_width', type=int, default=40, help='Width of the input')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of color channel')
    parser.add_argument('--train_proportion', type=float, default=0.8, help='Proportion for training data split')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden nodes (information bottleneck)')
    parser.add_argument('--output_size', type=int, default=1, help='Number of neurons for prediction')
    parser.add_argument('--conv3d_out_channels', type=int, default=10, help='Number of temporal in CNN3D')
    parser.add_argument('--conv2_out_channels', type=int, default=64, help='Number of output in 2nd convolution layer')
    parser.add_argument('--conv2_1st_layer_kernel', type=int, default=4,
                        help='Size of kernel in 1st layer of 2d convolution layer')
    parser.add_argument('--conv2_2nd_layer_kernel', type=int, default=5,
                        help='Size of kernel in 2nd layer of 2d convolution layer')
    # Training procedure
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/model.pth',
                        help='Path to save load model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Flag to load the model from checkpoint')
    parser.add_argument('--cache_size', type=int, default=100, help='Maximum number of 3d tensor loaded in the memory')
    # Data specificity (neuro dataset)
    parser.add_argument('--chunk_size', type=int, default=9, help='Number of continuous data point in one chunk')
    parser.add_argument('--data_stride', type=int, default=2, help='Number of step to create data (10 ms / per step)')
    parser.add_argument('--max_recording_cell', type=int, default=50, help='Maximum number of recording neuron (for bulk)')
    parser.add_argument('--image_loading_method', type=str, default='ph', help='The loading method (ph, png, hdf5)')
    # Perceiver specificity
    parser.add_argument('--num_head', type=int, default=4, help='Number of heads in perceiver')
    parser.add_argument('--num_iter', type=int, default=1, help='Number of input reiteration')
    parser.add_argument('--num_latent', type=int, default=16, help='Number of latent length (encoding)')
    parser.add_argument('--num_band', type=int, default=10, help='Number of bands in positional encoding')
    parser.add_argument('--use_bulk', action='store_true', help='Enable whole recording processing')
    parser.add_argument('--use_layer_norm', action='store_true', help='Enable layer normalization')
    parser.add_argument('--concatenate_positional_encoding', action='store_true',
                        help='Enable concatenation for positional encoding')
    parser.add_argument('--use_phase_shift', action='store_true',
                        help='Shift phase of individual frequency')
    parser.add_argument('--use_dense_frequency', action='store_true',
                        help='Use dense frequency generator')
    parser.add_argument('--kernel_size', nargs=3, type=int, default=[2, 2, 2],
                        help='Input kernel size as three separate integers. Default is (2, 2, 2)')
    parser.add_argument('--stride', nargs=3, type=int, default=[1, 1, 1],
                        help='Input stride as three separate integers. Default is (1, 1, 1)')
    parser.add_argument('--loss_fn', type=str, default='MSE', choices=list(loss_functions.keys()),
                        help='The name of the loss function to use (default: MSE)')
    parser.add_argument('--encoding_method', type=str, default='max_spacing', help='Choose the SeriesEncoder method ('
                                                                                   'max_spacing or uniform)')

    # System computing enhancement
    parser.add_argument('--parallel_processing', action='store_true', help='Enable parallel_processing')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Accumulate gradients')
    # Plot parameters
    parser.add_argument('--num_cols', type=int, default=5, help='Number of columns in a figure')
    parser.add_argument('--add_sampler', action='store_true', help='Enable efficient sampler for dataset')

    return parser.parse_args()


def main():
    # for running a new set of neurons, remember to change the neu_dir and
    args = parse_args()
    config_module = f"configs.neuros.{args.config_name}"
    config = __import__(config_module, fromlist=[''])
    filename_fixed = args.experiment_name
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    image_root_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/'
    link_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Link/'
    resp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Response/'
    savemat_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Matfiles/'
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
    torch.cuda.empty_cache()

    # num_workers = os.cpu_count()
    # mp.set_start_method('spawn', force=True)
    # logging.info(f'Number of workers: {num_workers} \n')
    logging.info(f'CUDA counts: {torch.cuda.device_count} \n')

    experiment_info_table = getattr(config, 'experiment_info_table', None)
    filtered_data = getattr(config, 'filtered_data', None)
    logging.info(f'filtered_data size:{filtered_data.shape} \n')
    logging.info(f'filtered_data:{filtered_data} \n')

    # construct the array for dataset
    data_constructor = DataConstructor(filtered_data, seq_len=args.input_depth, stride=args.data_stride,
                                       link_dir=link_dir, resp_dir=resp_dir, max_recording_cell=args.max_recording_cell)
    if args.use_bulk:
        data_array, query_array, query_index, firing_rate_array = data_constructor.construct_data_in_bulk()
        # query_index is a list for mapping experiment to query_array_ids
    else:
        data_array, query_array, query_index, firing_rate_array = data_constructor.construct_data()
        query_index = query_index.astype('int64')

    data_array = data_array.astype('int64')
    query_array = query_array.astype('int64')
    firing_rate_array = firing_rate_array.astype('float32')


    # Save to .mat file
    # filtered_data_mat = {col: filtered_data[col].values for col in filtered_data.columns}
    savemat_filename = f"{filename_fixed}_train_neuro_list.mat"
    savemat(os.path.join(savemat_dir, savemat_filename),
            {"data_array": data_array, "query_array": query_array,
             "query_index": query_index, "firing_rate_array": firing_rate_array})


    # construct the query array for query encoder
    query_df = pd.DataFrame(query_array, columns=['experiment_id', 'neuron_id'])
    query_array = pd.merge(query_df, experiment_info_table, on='experiment_id', how='left')
    query_array = query_array[['experiment_id', 'species_id', 'sex_id', 'neuron_id']]
    query_array['neuron_unique_id'] = query_array['experiment_id'] * 10000 + query_array['neuron_id']
    query_array = query_array.drop(['neuron_id'], axis=1)
    query_array = query_array.to_numpy()

    # Encode series_ids into query arrays
    query_encoder = SeriesEncoder(getattr(config, 'query_max_values', None),
                                  getattr(config, 'query_lengths', None),
                                  encoding_method=args.encoding_method,
                                  shuffle_components=getattr(config, 'query_shuffle_components', None))
    # query_encoder = SeriesEncoder(max_values, lengths, shuffle_components=shuffle_components)
    logging.info(f'(bef) query_array size:{query_array.shape} \n')
    logging.info(f'(bef) query_array:{query_array} \n')
    query_array = query_encoder.encode(query_array)
    logging.info(f'query_array size:{query_array.shape} \n')
    logging.info(f'query_array:{query_array} \n')

    # raise RuntimeError("Check query_array")
    # get data spit with chucks
    train_indices, val_indices = train_val_split(len(data_array), args.chunk_size, test_size=1-args.train_proportion)
    # get dataset
    train_dataset = RetinalDataset(data_array, query_index, firing_rate_array, image_root_dir, train_indices,
                                   args.chunk_size, device=device, cache_size=args.cache_size,
                                   image_loading_method=args.image_loading_method)
    train_loader = BatchGenerator(train_dataset, batch_size_target=args.batch_size, is_shuffle=True)
    val_dataset = RetinalDataset(data_array, query_index, firing_rate_array, image_root_dir, val_indices,
                                 args.chunk_size, device=device, cache_size=args.cache_size,
                                 image_loading_method=args.image_loading_method)
    val_loader = BatchGenerator(val_dataset, batch_size_target=args.batch_size, is_shuffle=False)

    ''' (debug) maynot work with use_bulk '''
    check_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    dataiter = iter(check_loader)
    movie, labels, index = next(dataiter)
    logging.info(f'movie clip: {movie.shape} labels:{labels} index:{index} \n')
    queryvec = torch.from_numpy(query_array).unsqueeze(1)
    queryvec = queryvec[index]
    logging.info(f'query vector: {queryvec.shape} \n')
    logging.info(f'query: {queryvec}\n')
    # plot and save the target_matrix figure
    plot3dmat(movie[0, 0, :, :, :].squeeze(), args.num_cols, savefig_dir, file_prefix='plot_3D_matrix')
    # Model, Loss, and Optimizer
    if args.model == 'RetinalPerceiver':
        model = RetinalPerceiverIO(input_dim=args.input_channels, latent_dim=args.hidden_size,
                                   output_dim=args.output_size,
                                   num_latents=args.num_latent, heads=args.num_head, depth=args.num_iter,
                                   query_dim=query_array.shape[1],
                                   depth_dim=args.input_depth, height=args.input_height, width=args.input_width,
                                   num_bands=args.num_band, use_layer_norm=args.use_layer_norm,
                                   kernel_size=args.kernel_size,
                                   stride=args.stride,
                                   concatenate_positional_encoding=args.concatenate_positional_encoding,
                                   use_phase_shift=args.use_phase_shift, use_dense_frequency=args.use_dense_frequency).to(device)
    elif args.model == 'RetinalCNN':
        model = RetinalPerceiverIOWithCNN(input_depth=args.input_depth, input_height=args.input_height,
                                          input_width=args.input_width, output_dim=args.output_size,
                                          latent_dim=args.hidden_size,
                                          query_dim=query_array.shape[1], num_latents=args.num_latent,
                                          heads=args.num_head,
                                          use_layer_norm=args.use_layer_norm, num_bands=args.num_band,
                                          conv3d_out_channels=args.conv3d_out_channels,
                                          conv2_out_channels=args.conv2_out_channels,
                                          conv2_1st_layer_kernel=args.conv2_1st_layer_kernel,
                                          conv2_2nd_layer_kernel=args.conv2_2nd_layer_kernel,
                                          device=device).to(device)

    if args.parallel_processing:
        model = nn.DataParallel(model)
        # model = DistributedDataParallel(model, device_ids=[local_rank])
        logging.info(f'Initial parallel processing on on {torch.cuda.device_count()} \n')

    logging.info(f'Model: {args.model} \n')
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    '''
    summary(model,
            input_data=(
                torch.rand(1, args.input_channels, args.input_depth, args.input_height, args.input_width).to(device),
                torch.rand(1, 1, query_array.shape[1]).to(device)
            ))
    '''
    sys.stdout = old_stdout
    logging.info(buffer.getvalue())

    criterion = loss_functions[args.loss_fn]
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Initialize the Trainer
    trainer = Trainer(model, criterion, optimizer, device, args.accumulation_steps,
                      query_array=query_array)
    # Initialize the Evaluator
    evaluator = Evaluator(model, criterion, device, query_array=query_array)

    # Optionally, load from checkpoint
    if args.load_checkpoint:
        checkpoint_loader = CheckpointLoader(checkpoint_path=args.checkpoint_path, device=device)
        model, optimizer = checkpoint_loader.load_checkpoint(model, optimizer)
        start_epoch = checkpoint_loader.get_epoch()
        training_losses, validation_losses = checkpoint_loader.get_training_losses(), checkpoint_loader.get_validation_losses()
    else:
        start_epoch = 0
        training_losses = []
        validation_losses = []
        start_time = time.time()  # Capture the start time

    for epoch in range(start_epoch, args.epochs):
        avg_train_loss = trainer.train_one_epoch(train_loader)
        training_losses.append(avg_train_loss)

        # torch.cuda.empty_cache()
        avg_val_loss = evaluator.evaluate(val_loader)
        validation_losses.append(avg_val_loss)

        # Print training status
        if (epoch + 1) % 5 == 0:
            elapsed_time = time.time() - start_time
            # Log the epoch and elapsed time, and on a new indented line, log the losses
            logging.info(
                f"{filename_fixed} Epoch [{epoch + 1}/{args.epochs}], Elapsed time: {elapsed_time:.2f} seconds \n"
                f"\tTraining Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f} \n")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:  # Example: Save every 10 epochs
            checkpoint_filename = f'{filename_fixed}_checkpoint_epoch_{epoch + 1}.pth'
            logging.info(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB \n"
                         f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e6} MB \n")

            # Assert statements to check that neither variable is None or undefined
            assert training_losses is not None, "training_losses is None or undefined"
            assert validation_losses is not None, "validation_losses is None or undefined"

            save_checkpoint(epoch, model, optimizer, args, training_losses, validation_losses,
                            file_path=os.path.join(savemodel_dir, checkpoint_filename))


if __name__ == '__main__':
    main()
