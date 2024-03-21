import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import numpy as np
import logging
import time
import pandas as pd
import gc
import psutil

# from io import StringIO
# import sys
# from torchinfo import summary
# from scipy.io import savemat
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel

from datasets.neuron_dataset import RetinalDataset, DataConstructor
from datasets.neuron_dataset import train_val_split, load_data_from_excel, filter_and_merge_data
from utils.utils import plot_and_save_3d_matrix_with_timestamp as plot3dmat
from models.perceiver3d import RetinalPerceiverIO
from models.cnn3d import RetinalPerceiverIOWithCNN
from models.style3d import StyleCNN
from models.FiLM3d import FiLMCNN
from utils.training_procedure import Trainer, Evaluator, save_checkpoint, CheckpointLoader
from utils.loss_function import loss_functions


def get_cpu_free_memory():
    memory = psutil.virtual_memory()
    free_memory_bytes = memory.free  # Free memory in bytes
    return free_memory_bytes


def get_gpu_free_memory(device_id=0):
    torch.cuda.synchronize(device_id)
    torch.cuda.empty_cache()  # Clear any cached memory to get a more accurate reading
    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    allocated_memory = torch.cuda.memory_allocated(device_id)
    free_memory = total_memory - allocated_memory
    return free_memory


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
    parser.add_argument('--model', type=str, choices=['RetinalPerceiver', 'RetinalCNN', 'AdaptiveCNN', 'FiLMCNN'],
                        required=True, help='Model to train')
    parser.add_argument('--input_depth', type=int, default=20, help='Number of time points')
    parser.add_argument('--input_height', type=int, default=30, help='Heights of the input')
    parser.add_argument('--input_width', type=int, default=40, help='Width of the input')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of color channel')
    parser.add_argument('--train_proportion', type=float, default=0.8, help='Proportion for training data split')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden nodes (information bottleneck)')
    parser.add_argument('--output_size', type=int, default=1, help='Number of neurons for prediction')
    parser.add_argument('--conv3d_out_channels', type=int, default=10, help='Number of temporal in CNN3D')
    parser.add_argument('--conv2_out_channels', type=int, default=64, help='Number of output in 2nd convolution layer')
    parser.add_argument('--conv2_1st_layer_kernel', type=int, default=3,
                        help='Size of kernel in 1st layer of 2d convolution layer')
    parser.add_argument('--conv2_2nd_layer_kernel', type=int, default=3,
                        help='Size of kernel in 2nd layer of 2d convolution layer')
    parser.add_argument('--conv2_3rd_layer_kernel', type=int, default=3,
                        help='Size of kernel in 3rd layer of 2d convolution layer')
    parser.add_argument('--momentum', type=float, default=0.1, help='Window for moving batch normalization')
    # embedding size
    parser.add_argument('--num_dataset', type=int, default=10, help='Number of potential experiments')
    parser.add_argument('--num_neuron', type=int, default=100, help='Number of encoding neurons')
    parser.add_argument('--dataset_embedding_length', type=int, default=5, help='Length of embedding for dataset ids')
    parser.add_argument('--neuronid_embedding_length', type=int, default=16, help='Length of embedding for neuron ids')
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
    parser.add_argument('--image_loading_method', type=str, default='ph', help='The loading method (ph, png, hdf5)')
    # Perceiver specificity
    parser.add_argument('--num_head', type=int, default=4, help='Number of heads in perceiver')
    parser.add_argument('--num_iter', type=int, default=1, help='Number of input reiteration')
    parser.add_argument('--num_latent', type=int, default=16, help='Number of latent length (encoding)')
    parser.add_argument('--num_band', type=int, default=10, help='Number of bands in positional encoding')
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
    parser.add_argument('--is_selective_layers', action='store_true', help='Enable L1 penalty for feature and spatial selection')
    parser.add_argument('--is_feature_L1', action='store_true',
                        help='Enable L1 penalty for FiLM type for attention module')
    parser.add_argument('--lambda_l1', type=float, default=0.01, help='L1 weight penalty for selective layers')

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
    torch.backends.cudnn.benchmark = False
    device_id = 0
    torch.cuda.set_device(device_id)

    # num_workers = os.cpu_count()
    # mp.set_start_method('spawn', force=True)
    # logging.info(f'Number of workers: {num_workers} \n')
    logging.info(f'CUDA counts: {torch.cuda.device_count} \n')

    experiment_session_table = getattr(config, 'experiment_session_table', None)
    included_neuron_table = getattr(config, 'included_neuron_table', None)
    experiment_info_table = getattr(config, 'experiment_info_table', None)
    experiment_neuron_table = getattr(config, 'experiment_neuron_table', None)
    filtered_data = filter_and_merge_data(
        experiment_session_table, experiment_neuron_table,
        selected_experiment_ids=[1],
        selected_stimulus_types=[1, 2, 3],
        excluded_session_table=None,
        excluded_neuron_table=None,
        included_session_table=None,
        included_neuron_table=included_neuron_table
    )
    logging.info(f'filtered_data size:{filtered_data.shape} \n')
    logging.info(f'filtered_data:{filtered_data} \n')

    # construct the array for dataset
    data_constructor = DataConstructor(filtered_data, seq_len=args.input_depth, stride=args.data_stride,
                                       link_dir=link_dir, resp_dir=resp_dir)
    data_array, query_array, query_index, firing_rate_array = data_constructor.construct_data()
    data_array = data_array.astype('int64')
    query_array = query_array.astype('int64')
    query_index = query_index.astype('int64')
    firing_rate_array = firing_rate_array.astype('float32')

    '''
    # Save to .mat file
    # filtered_data_mat = {col: filtered_data[col].values for col in filtered_data.columns}
    savemat(os.path.join(savemat_dir, 'train_neuro_list.mat'),
            {"data_array": data_array, "query_array": query_array,
             "query_index": query_index, "firing_rate_array": firing_rate_array})
    raise RuntimeError("Script stopped after saving outputs.")
    '''

    # construct the query array for query encoder
    query_df = pd.DataFrame(query_array, columns=['experiment_id', 'neuron_id'])
    query_array = pd.merge(query_df, experiment_info_table, on='experiment_id', how='left')
    query_array = query_array[['experiment_id', 'species_id', 'sex_id', 'neuron_id']]
    query_array['neuron_unique_id_unsorted'] = query_array['experiment_id'] * 1000 + query_array['neuron_id']
    query_array['neuron_unique_id'] = pd.factorize(query_array['neuron_unique_id_unsorted'])[0]
    query_array['experiment_unique_id'] = pd.factorize(query_array['experiment_id'])[0]
    query_array = query_array.drop(['neuron_id', 'neuron_unique_id_unsorted', 'experiment_id'], axis=1)
    query_array = query_array[['experiment_unique_id', 'species_id', 'sex_id', 'neuron_unique_id']]
    query_array = query_array.to_numpy()

    del experiment_session_table, included_neuron_table, experiment_info_table, experiment_neuron_table
    del query_df, data_constructor, filtered_data

    logging.info(f'query_array size:{query_array.shape} \n')
    logging.info(f'query_array:{query_array} \n')

    # get data spit with chucks
    train_indices, val_indices = train_val_split(len(data_array), args.chunk_size, test_size=1-args.train_proportion)
    # get dataset
    train_dataset = RetinalDataset(data_array, query_index, firing_rate_array, image_root_dir, train_indices,
                                   args.chunk_size, device=device, cache_size=args.cache_size,
                                   image_loading_method=args.image_loading_method)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = RetinalDataset(data_array, query_index, firing_rate_array, image_root_dir, val_indices,
                                 args.chunk_size, device=device, cache_size=args.cache_size,
                                 image_loading_method=args.image_loading_method)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # show one example
    dataiter = iter(train_loader)
    movie, labels, index = next(dataiter)
    logging.info(f'movie clip: {movie.shape} labels:{labels} index:{index} \n')
    # plot and save the target_matrix figure
    plot3dmat(movie[0, 0, :, :, :].squeeze(), args.num_cols, savefig_dir, file_prefix='plot_3D_matrix')

    dummy_query_array = torch.from_numpy(query_array)
    dummy_query_vectors = dummy_query_array[index]
    dummy_dataset_ids = dummy_query_vectors[:, 0]
    dummy_neuron_ids = dummy_query_vectors[:, 3]
    logging.info(f'dummy_dataset_ids: {dummy_dataset_ids} \n')
    logging.info(f'dummy_neuron_ids: {dummy_neuron_ids} \n')
    del dummy_query_array, dummy_query_vectors, dummy_dataset_ids, dummy_neuron_ids
    del movie, labels, index, dataiter
    # Model, Loss, and Optimizer
    if args.model == 'AdaptiveCNN':
        model = StyleCNN(input_depth=args.input_depth, input_height=args.input_height, input_width=args.input_width,
                         conv3d_out_channels=args.conv3d_out_channels, conv2_out_channels=args.conv2_out_channels,
                         conv2_1st_layer_kernel=args.conv2_1st_layer_kernel, conv2_2nd_layer_kernel=args.conv2_2nd_layer_kernel,
                         conv2_3rd_layer_kernel=args.conv2_3rd_layer_kernel, num_dataset=args.num_dataset,
                         momentum=args.momentum, num_neuron=args.num_neuron).to(device)
    elif args.model == 'FiLMCNN':
        model = FiLMCNN(input_depth=args.input_depth, input_height=args.input_height, input_width=args.input_width,
                         conv3d_out_channels=args.conv3d_out_channels, conv2_out_channels=args.conv2_out_channels,
                         conv2_1st_layer_kernel=args.conv2_1st_layer_kernel, conv2_2nd_layer_kernel=args.conv2_2nd_layer_kernel,
                         conv2_3rd_layer_kernel=args.conv2_3rd_layer_kernel, num_dataset=args.num_dataset,
                         dataset_embedding_length=args.dataset_embedding_length, num_neuron=args.num_neuron,
                         neuronid_embedding_length=args.neuronid_embedding_length, momentum=args.momentum).to(device)
    elif args.model == 'RetinalPerceiver':
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

    '''
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    sys.stdout = old_stdout
    logging.info(buffer.getvalue())
    '''

    criterion = loss_functions[args.loss_fn]
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Initialize the Trainer
    trainer = Trainer(model, criterion, optimizer, device, args.accumulation_steps, is_feature_L1=args.is_feature_L1,
                      query_array=query_array, is_selective_layers=args.is_selective_layers, lambda_l1=args.lambda_l1)
    logging.info('Trainer is loaded \n')
    # Initialize the Evaluator
    evaluator = Evaluator(model, criterion, device, query_array=query_array, is_feature_L1=args.is_feature_L1,
                          is_selective_layers=args.is_selective_layers, lambda_l1=args.lambda_l1)
    logging.info('Evaluator is loaded \n')
    # Optionally, load from checkpoint
    if args.load_checkpoint:
        checkpoint_loader = CheckpointLoader(checkpoint_path=args.checkpoint_path, device=device)
        model, optimizer = checkpoint_loader.load_checkpoint(model, optimizer)
        start_epoch = checkpoint_loader.get_epoch()
        training_losses = checkpoint_loader.get_training_losses()
        validation_losses = checkpoint_loader.get_validation_losses()
    else:
        start_epoch = 0
        training_losses = []
        validation_losses = []
        start_time = time.time()  # Capture the start time
        logging.info('No checkpoint \n')

    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        gc.collect()
        avg_train_loss = trainer.train_one_epoch(train_loader)
        training_losses.append(avg_train_loss)

        # torch.cuda.empty_cache()
        # gc.collect()
        avg_val_loss = evaluator.evaluate(val_loader)
        validation_losses.append(avg_val_loss)
        # logging.info(f'epoch (validation): {epoch} \n')
        # logging.info(f"CPU free memory: {get_cpu_free_memory() / 1e6} MB \n"
        #              f"GPU from memory: {get_gpu_free_memory(device_id) / 1e6} MB \n")
        # memory_summary = torch.cuda.memory_summary(abbreviated=False)
        # logging.info(f"GPU monitoring: \n {memory_summary} \n")
        # memory_summary = torch.cuda.memory_summary(device, abbreviated=False)
        # logging.info(f"specific GPU monitoring: \n { memory_summary}  \n")
        # logging.info(f"\tTraining Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f} \n")
        # logging.info(f"GPU Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB \n"
        #             f"GPU Max memory allocated: {torch.cuda.max_memory_allocated() / 1e6} MB \n")
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
