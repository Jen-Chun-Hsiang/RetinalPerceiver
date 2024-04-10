import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from datetime import datetime
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from scipy.io import savemat

from utils.training_procedure import CheckpointLoader, forward_model
from datasets.simulated_dataset import MultiMatrixDataset
from datasets.neuron_dataset import RetinalDataset, DataConstructor
from datasets.neuron_dataset import train_val_split, load_mat_to_dataframe, load_data_from_excel, filter_and_merge_data
from utils.utils import DataVisualizer, SeriesEncoder
from utils.utils import calculate_correlation
from models.classic3d import Classic3d


def main():
    stimulus_type = 'cnn_2024040901_GoodCell1'  # get the name from the check point folder
    epoch_end = 200  # the number of epoch in the check_point file
    total_length = 10000
    initial_size = (10, 24, 32)
    is_original_dataset = False  # use original training data (True) or use the white noise generator (False)
    is_encoding_query = False  # whether SeriesEncode was applied (or default embedding)
    is_weight_in_label = False  # check if the data is good
    is_full_figure_draw = True  # determine whether draw for each neuro or just get stats
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    savedata_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Data/'
    image_root_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/'
    checkpoint_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    exp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/ExperimentSheets.xlsx'
    neu_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/experiment_neuron_021324.mat'
    link_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Link/'
    resp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Response/'
    mat_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Matfiles/'

    # Compile the regarding parameters
    checkpoint_filename = f'PerceiverIO_{stimulus_type}_checkpoint_epoch_{epoch_end}'
    checkpoint_path = os.path.join(checkpoint_folder, f'{checkpoint_filename}.pth')

    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(saveprint_dir, f'{checkpoint_filename}_training_log_{timestr}.txt')
    savedata_filename_npy = os.path.join(savedata_dir, f'{checkpoint_filename}_data.npy')
    savedata_filename_mat = os.path.join(savedata_dir, f'{checkpoint_filename}_data.mat')
    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    #
    visualizer_prog = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Training_progress')

    # Load the training and model parameters
    checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path, device=device)
    training_losses, validation_losses = checkpoint_loader.load_training_losses(), checkpoint_loader.load_validation_losses()
    logging.info(f'training_losses:{training_losses} \n')
    logging.info(f'validation_losses:{validation_losses} \n')
    visualizer_prog.plot_and_save(None, plot_type='line', line1=training_losses, line2=validation_losses,
                                  xlabel='Epochs', ylabel='Loss')
    args = checkpoint_loader.load_args()
    config_module = f"configs.neuros.{args.config_name}"
    config = __import__(config_module, fromlist=[''])

    # In the upcoming training procedure, the tables below will be saved with the training file
    # Load and make sure the table is correct

    experiment_session_table = getattr(config, 'experiment_session_table', None)
    included_neuron_table = getattr(config, 'included_neuron_table', None)
    experiment_info_table = getattr(config, 'experiment_info_table', None)
    experiment_neuron_table = getattr(config, 'experiment_neuron_table', None)

    filtered_data = filter_and_merge_data(
        experiment_session_table, experiment_neuron_table,
        selected_experiment_ids=[1],
        selected_stimulus_types=[2],
        excluded_session_table=None,
        excluded_neuron_table=None,
        included_session_table=None,
        included_neuron_table=included_neuron_table
    )

    # construct the array for dataset
    data_constructor = DataConstructor(filtered_data, seq_len=args.input_depth, stride=args.data_stride,
                                       link_dir=link_dir, resp_dir=resp_dir)
    data_array, query_array, query_index, firing_rate_array = data_constructor.construct_data()
    data_array = data_array.astype('int64')
    query_array = query_array.astype('int64')
    query_index = query_index.astype('int64')
    firing_rate_array = firing_rate_array.astype('float32')

    del experiment_session_table, included_neuron_table, experiment_info_table, experiment_neuron_table
    del data_constructor, filtered_data

    # Get how many unique cells are there
    train_indices, val_indices = train_val_split(len(data_array), args.chunk_size, test_size=1 - args.train_proportion)
    train_dataset = RetinalDataset(data_array, query_index, firing_rate_array, image_root_dir, train_indices,
                                   args.chunk_size, device=device, cache_size=args.cache_size,
                                   image_loading_method=args.image_loading_method)

    check_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    dataiter = iter(check_loader)
    movie, labels, index = next(dataiter)
    logging.info(f'movie clip: {movie.shape} labels:{labels} index:{index} \n')
    del movie, labels, index, dataiter, check_loader

    # Initialize the DataVisualizer
    visualizer_est_rf = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Estimate_RF')
    visualizer_est_rfstd = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Estimate_RF_std')
    visualizer_inout_corr = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Input_output_correlation')

    if args.model == 'ClassicCNN':
        model = Classic3d(input_depth=args.input_depth, input_height=args.input_height, input_width=args.input_width,
                          conv3d_out_channels=args.conv3d_out_channels, conv2_out_channels=args.conv2_out_channels,
                          conv2_1st_layer_kernel=args.conv2_1st_layer_kernel,
                          conv2_2nd_layer_kernel=args.conv2_2nd_layer_kernel,
                          conv2_3rd_layer_kernel=args.conv2_3rd_layer_kernel).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model, optimizer = checkpoint_loader.load_checkpoint(model, optimizer)

    num_cols = 5

    sample_data, sample_label, sample_index = train_dataset[0]
    logging.info(f"dataset size: {sample_data.shape}")
    # if specified query array, always make sure is_weight_in_label
    if is_original_dataset:
        dataset_test = train_dataset
    else:
        dataset_test = MultiMatrixDataset(sample_data, length=total_length, device=device, combination_set=[1],
                                          initial_size=initial_size)
    output_image, weights, labels = forward_model(model, dataset_test, batch_size=8, use_matrix_index=False,
                                                  is_weight_in_label=is_weight_in_label, model_type=args.model)

    if is_full_figure_draw:
        output_image_np = output_image.squeeze().cpu().numpy()
        visualizer_est_rf.plot_and_save(output_image_np, plot_type='3D_matrix', num_cols=5)
        visualizer_inout_corr.plot_and_save(None, plot_type='scatter', x_data=labels, y_data=weights,
                                            xlabel='Labels', ylabel='Weights',
                                            title='Relationship between Weights and Labels')
        output_image_np_std = np.std(output_image_np, axis=0)
        visualizer_est_rfstd.plot_and_save(output_image_np_std, plot_type='2D_matrix')

    corrcoef_vals = calculate_correlation(labels, weights)

    logging.info(f'correlation coefficient: {corrcoef_vals} \n')
    np.save(savedata_filename_npy, corrcoef_vals)

    # Save the dictionary as a .mat file
    savemat(savedata_filename_mat, {'corrcoef_vals': corrcoef_vals})


if __name__ == "__main__":
    main()
