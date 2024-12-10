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
import random
import torch.distributed as dist
# from torchinfo import summary
# from scipy.io import savemat
# from torch.nn.parallel import DistributedDataParallel
# from utils.query_editor import get_unique_sets, QueryPermutator

from datasets.simulated_target_rf import MultiTargetMatrixGenerator, ParameterGenerator
from utils.utils import plot_and_save_3d_matrix_with_timestamp as plot3dmat
from utils.utils import SeriesEncoder
from utils.accessory import calculate_mask_positions
from datasets.simulated_dataset import MultiMatrixDataset
from models.perceiver3d import RetinalPerceiverIO
from models.cnn3d import RetinalPerceiverIOWithCNN
from utils.training_procedure import Trainer, Evaluator, save_checkpoint, CheckpointLoader
from utils.value_inspector import save_distributions


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
    parser.add_argument('--config_name', type=str, default='sim_022822', help='Config file name for data generation')
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')
    parser.add_argument('--rng_seed', type=int, default=None, help='assign a random seed')
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
    parser.add_argument('--conv2_out_channels', type=int, default=64, help='Number of output in 2nd convolution layer')
    parser.add_argument('--conv2_1st_layer_kernel', type=int, default=3,
                        help='Size of kernel in 1st layer of 2d convolution layer')
    parser.add_argument('--conv2_2nd_layer_kernel', type=int, default=3,
                        help='Size of kernel in 2nd layer of 2d convolution layer')
    # Training procedure
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--schedule_method', type=str, default='RLRP', help='Method used for scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--schedule_factor', type=float, default=0.2, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/model.pth',
                        help='Path to save load model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Flag to load the model from checkpoint')
    parser.add_argument('--masking_pos', type=int, nargs='+', default=None, help='masking positions, such as (0, 1, 2)')
    parser.add_argument('--masking_prob', type=float, default=0.5, help='Probability of masking')
    # Target matrix specificity
    parser.add_argument('--sf_surround_weight', type=float, default=0.5, help='Strength of spatial surround')
    parser.add_argument('--tf_surround_weight', type=float, default=0.2, help='Strength of temporal surround')
    parser.add_argument("--mean", nargs=2, type=float, default=(0.1, -0.2),
                        help="Mean as two separate floats (e.g., 0.1 -0.2)")
    parser.add_argument("--mean2", nargs=2, type=float, default=None,
                        help="Mean as two separate floats (e.g., 0.1 -0.2)")
    parser.add_argument("--cov", type=parse_covariance, default=np.array([[0.12, 0.05], [0.04, 0.03]]),
                        help="Covariance matrix as four floats separated by commas (e.g., '0.12,0.05,0.04,0.03')")
    parser.add_argument("--cov2", type=parse_covariance, default=None,
                        help="Covariance matrix as four floats separated by commas (e.g., '0.12,0.05,0.04,0.03')")
    parser.add_argument('--add_noise', action='store_true', help='Enable adding noise to the output label')
    parser.add_argument('--noise_level', type=float, default=0.01, help='std of the noise, if added')
    parser.add_argument('--use_relu', action='store_true', help='Rectify the output label')
    parser.add_argument('--output_offset', type=float, default=0.01, help='add value to offset for rectification')
    # Stimulus specificity
    parser.add_argument('--stimulus_type', type=int, default=4, help='Stimulus type')
    parser.add_argument('--stimulus_type_set', nargs='+', type=int, default=[1], help='Sets of stimulus type')
    parser.add_argument('--is_norm_matrix', action='store_true', help='Normalize the matrix for generating signal')
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
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for contrastive learning')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive learning')
    parser.add_argument('--contrastive_factor', type=float, default=0.01, help='Temperature for contrastive learning')
    parser.add_argument('--encoding_method', type=str, default='max_spacing', help='Choose the SeriesEncoder method ('
                                                                                   'max_spacing or uniform)')
    # System computing enhancement
    parser.add_argument('--parallel_processing', action='store_true', help='Enable parallel_processing')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Accumulate gradients')
    parser.add_argument('--is_contrastive_learning', action='store_true', help='Enable contrastive learning')
    parser.add_argument('--num_worker', type=int, default=0, help='Use to offline loading data in batch')
    parser.add_argument('--do_not_train', action='store_true', help='Only present the values without training')
    parser.add_argument('--is_GPU', action='store_true', help='Using GPUs for accelaration')
    parser.add_argument('--exam_batch_idx', type=int, default=None,
                        help='examine the timer and stop code in the middle')
    parser.add_argument('--timer_tau', type=float, default=0.99, help='Set timer tau')
    parser.add_argument('--timer_n', type=int, default=200, help='Set timer n')
    # Plot parameters
    parser.add_argument('--num_cols', type=int, default=5, help='Number of columns in a figure')

    return parser.parse_args()


def main():
    args = parse_args()
    config_module = f"configs.sims.{args.config_name}"
    config = __import__(config_module, fromlist=[''])

    filename_fixed = args.experiment_name
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    savemat_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Matfiles/'
    # Generate a timestamp
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Construct the full path for the log file
    log_filename = os.path.join(saveprint_dir, f'{filename_fixed}_training_log_{timestr}.txt')

    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f'start logging... \n')

    if args.is_GPU:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        logging.info(f'set up GPU operation \n')
    else:
        device = 'cpu'
        logging.info(f'set up CPU operation \n')

    if args.rng_seed is None:
        args.rng_seed = random.randint(0, int(1e9))

    query_table = getattr(config, 'query_table', None)
    sf_param_table = getattr(config, 'sf_param_table', None)
    tf_param_table = getattr(config, 'tf_param_table', None)
    logging.info(f'query_table: {query_table} \n')
    # Generate param_list
    parameter_generator = ParameterGenerator(sf_param_table, tf_param_table, seed=args.rng_seed)
    param_lists, series_ids = parameter_generator.generate_parameters(query_table)
    # param_list, series_ids = integrated_list.generate_combined_param_list()
    # logging.info(f'param_list: {param_list} \n')
    # logging.info(f'series_ids: {series_ids} \n')

    # Encode series_ids into query arrays
    max_values = getattr(config, 'max_values', None)
    # skip_encoding = getattr(config, 'skip_encoding', None)
    encoding_type = getattr(config, 'encoding_type', None)
    lengths = getattr(config, 'lengths', None)
    shuffle_components = getattr(config, 'shuffle_components', None)
    query_encoder = SeriesEncoder(max_values, lengths, encoding_method=args.encoding_method, encoding_type=encoding_type,
                                  shuffle_components=shuffle_components)
    query_array = query_encoder.encode(series_ids)
    logging.info(f'query_array size:{query_array.shape} \n')
    logging.info(f'query_array:{query_array[:3]} \n')

    query_permutator = None
    '''
    # Save to .mat file
    savemat(os.path.join(savemat_dir, 'sim_multi_list.mat'),
            {"param_list": param_list, "series_ids": series_ids, 'query_array': query_array})
    raise RuntimeError("Script stopped after saving outputs.")
    '''

    # Use param_list in MultiTargetMatrixGenerator
    multi_target_gen = MultiTargetMatrixGenerator(param_lists, is_norm_matrix=args.is_norm_matrix)
    print(f'height: {args.input_height}')
    print(f'width: {args.input_width}')
    print(f'depth: {args.input_depth}')
    target_matrix = multi_target_gen.create_3d_target_matrices(
        input_height=args.input_height, input_width=args.input_width, input_depth=args.input_depth)

    logging.info(f'target matrix: {target_matrix.shape}  \n')
    # plot and save the target_matrix figure
    plot3dmat(target_matrix[0, :, :, :], args.num_cols, savefig_dir, file_prefix='plot_3D_matrix')

    # Initialize the dataset
    dataset = MultiMatrixDataset(target_matrix, length=args.total_length, combination_set=args.stimulus_type_set,
                                 add_noise=args.add_noise, noise_level=args.noise_level, use_relu=args.use_relu,
                                 output_offset=args.output_offset)

    # Splitting the dataset into training and validation sets
    train_length = int(0.8 * args.total_length)  # 80% for training
    val_length = args.total_length - train_length  # 20% for validation

    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
    if args.num_worker == 0:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_worker, pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_worker, pin_memory=True, persistent_workers=False)

    check_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    dataiter = iter(check_loader)
    movie, labels, index = next(dataiter)
    queryvec = torch.from_numpy(query_array).unsqueeze(1)
    queryvec = queryvec[index]
    logging.info(f'movie clip: {movie.shape} labels:{labels} index:{index} \n')
    logging.info(f'query vector: {queryvec.shape} \n')
    # Model, Loss, and Optimizer
    if args.model == 'RetinalPerceiver':
        model = RetinalPerceiverIO(input_dim=args.input_channels, latent_dim=args.hidden_size,
                                   output_dim=args.output_size, num_latents=args.num_latent, heads=args.num_head,
                                   depth=args.num_iter, query_dim=query_array.shape[1], depth_dim=args.input_depth,
                                   height=args.input_height, width=args.input_width, num_bands=args.num_band,
                                   use_layer_norm=args.use_layer_norm, kernel_size=args.kernel_size, stride=args.stride,
                                   concatenate_positional_encoding=args.concatenate_positional_encoding,
                                   use_phase_shift=args.use_phase_shift,
                                   use_dense_frequency=args.use_dense_frequency).to(device)
    elif args.model == 'RetinalCNN':
        model = RetinalPerceiverIOWithCNN(input_depth=args.input_depth, input_height=args.input_height,
                                          input_width=args.input_width, output_dim=args.output_size,
                                          latent_dim=args.hidden_size, query_dim=query_array.shape[1],
                                          num_latents=args.num_latent, heads=args.num_head,
                                          use_layer_norm=args.use_layer_norm, num_bands=args.num_band,
                                          conv3d_out_channels=args.conv3d_out_channels,
                                          conv2_out_channels=args.conv2_out_channels,
                                          conv2_1st_layer_kernel=args.conv2_1st_layer_kernel,
                                          conv2_2nd_layer_kernel=args.conv2_2nd_layer_kernel).to(device)

    if args.parallel_processing:
        model = nn.DataParallel(model)
        # model = DistributedDataParallel(model, device_ids=[local_rank])
        logging.info(f'Initial parallel processing on on {torch.cuda.device_count()} \n')

    logging.info(f'Model: {args.model} \n')
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    '''
    summary(model,
            input_data=(torch.rand(1, args.input_channels, args.input_depth, args.input_height, args.input_width),
                        torch.rand(1, 1, query_array.shape[1])))
    '''
    sys.stdout = old_stdout
    logging.info(buffer.getvalue())

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.schedule_method.lower() == 'rlrp':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.schedule_factor, patience=5)
    elif args.schedule_method.lower() == 'cawr':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    # Initialize the Trainer
    if args.masking_pos is None:
        masking_pos = None
    else:
        masking_pos = calculate_mask_positions(lengths, args.masking_pos)
    trainer = Trainer(model, criterion, optimizer, device, args.accumulation_steps,
                      query_array=query_array, is_contrastive_learning=args.is_contrastive_learning,
                      series_ids=series_ids, query_encoder=query_encoder, query_permutator=query_permutator,
                      margin=args.margin, temperature=args.temperature, contrastive_factor=args.contrastive_factor,
                      masking_pos=masking_pos, masking_prob=args.masking_prob, timer_tau=args.timer_tau,
                      timer_n=args.timer_n)
    # Initialize the Evaluator
    evaluator_contra = Evaluator(model, criterion, device, query_array=query_array,
                                 is_contrastive_learning=args.is_contrastive_learning,
                                 series_ids=series_ids, query_permutator=query_permutator, query_encoder=query_encoder,
                                 margin=args.margin, temperature=args.temperature,
                                 contrastive_factor=args.contrastive_factor)
    evaluator = Evaluator(model, criterion, device, query_array=query_array)

    # Optionally, load from checkpoint
    if args.load_checkpoint:
        checkpoint_loader = CheckpointLoader(checkpoint_path=args.checkpoint_path)
        model, optimizer = checkpoint_loader.load_checkpoint(model, optimizer)
        start_epoch = checkpoint_loader.get_epoch()
        training_losses, validation_losses = checkpoint_loader.get_training_losses(), checkpoint_loader.get_validation_losses()
        learning_rate_dynamics = checkpoint_loader.get_learning_rate_dynamics()
    else:
        start_epoch = 0
        training_losses = []
        validation_losses = []
        learning_rate_dynamics = []
        validation_contra_losses = []
        start_time = time.time()  # Capture the start time

    if args.do_not_train:
        model.eval()  # Set the model to evaluation mode
        n = 1000
        plot_file_name = f'{filename_fixed}value_distribution_n{n}.png'
        save_distributions(train_loader, n=n, folder_name=savefig_dir, file_name=plot_file_name)


        # with torch.no_grad():  # Disable gradient computation
        #     for batch_idx, (random_matrix, output_value, _) in enumerate(train_loader):
        #         if batch_idx >= n:
        #             break  # Exit after processing n batches
        #
        #         # Print inputs and outputs for the current batch
        #         print(f"Batch {batch_idx + 1} Inputs:")
        #         print(f'random_matrix min {torch.min(random_matrix)}')
        #         print(f'random_matrix max {torch.max(random_matrix)}')
        #         print(f'output_value min {torch.min(output_value)}')
        #         print(f'output_value max {torch.max(output_value)}')
        #
        #         # outputs = model(sequences)
        #         # print(f"\nBatch {batch_idx + 1} Outputs:")
        #         # print(f'output min {torch.min(outputs)}')
        #         # print(f'output max {torch.max(outputs)}')
        #         print("\n" + "-" * 50 + "\n")

    else:
        for epoch in range(start_epoch, args.epochs):
            avg_train_loss = trainer.train_one_epoch(train_loader)
            training_losses.append(avg_train_loss)

            if args.exam_batch_idx is not None:
                logging.info(f'end at epoch: {epoch} \n')
                break

            # torch.cuda.empty_cache()
            avg_val_loss = evaluator.evaluate(val_loader)
            # scheduler.step(avg_val_loss)
            scheduler.step(epoch + (epoch / args.epochs))
            learning_rate_dynamics.append(scheduler.get_last_lr())
            validation_losses.append(avg_val_loss)
            avg_val_loss = evaluator_contra.evaluate(val_loader)
            validation_contra_losses.append(avg_val_loss)

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
                save_checkpoint(epoch, model, optimizer, scheduler, args, training_losses, validation_losses,
                                validation_contra_losses,
                                file_path=os.path.join(savemodel_dir, checkpoint_filename))

        if args.parallel_processing:
            # Clean up
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
