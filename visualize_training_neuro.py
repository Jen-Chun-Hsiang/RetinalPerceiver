import os
from datetime import datetime
import logging
import torch
import torch.optim as optim
import pandas as pd

from utils.training_procedure import CheckpointLoader, forward_model
from datasets.neuron_dataset import RetinalDataset, DataConstructor
from datasets.neuron_dataset import train_val_split, load_mat_to_dataframe, load_data_from_excel, filter_and_merge_data
from utils.utils import DataVisualizer, SeriesEncoder
from torch.utils.data import DataLoader, random_split
from models.perceiver3d import RetinalPerceiverIO
from models.cnn3d import RetinalPerceiverIOWithCNN
# (0) identify the cell we have modeled their responses
# (1) show the receptive field with the white noise
# (2) show the response to the test set


def main():
    stimulus_type = '50tpcnn_2024020601_GoodCell'  # get the name from the check point folder
    epoch_end = 100  # the number of epoch in the check_point file
    is_full_figure_draw = True  # determine whether draw for each neuro or just get stats
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    image_root_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/'
    checkpoint_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    exp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/ExperimentSheets.xlsx'
    neu_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/experiment_neuron_011724.mat'
    link_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Link/'
    resp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Response/'

    # Compile the regarding parameters
    checkpoint_filename = f'PerceiverIO_{stimulus_type}_checkpoint_epoch_{epoch_end}'
    checkpoint_path = os.path.join(checkpoint_folder, f'{checkpoint_filename}.pth')

    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(saveprint_dir, f'{checkpoint_filename}_training_log_{timestr}.txt')
    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
    device = torch.device("cuda")

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

    # In the upcoming training procedure, the tables below will be saved with the training file
    # Load and make sure the table is correct
    experiment_session_table = load_data_from_excel(exp_dir, 'experiment_session')
    experiment_session_table = experiment_session_table.drop('stimulus_type', axis=1)

    included_neuron_table = load_data_from_excel(exp_dir, 'included_neuron_01')

    experiment_info_table = load_data_from_excel(exp_dir, 'experiment_info')
    experiment_info_table = experiment_info_table.drop(['species', 'sex', 'day', 'folder_name'], axis=1)

    experiment_neuron_table = load_mat_to_dataframe(neu_dir, 'experiment_neuron_table', 'column_name')
    experiment_neuron_table.iloc[:, 0:3] = experiment_neuron_table.iloc[:, 0:3].astype('int64')

    experiment_neuron_table.fillna(-1, inplace=True)
    experiment_neuron_table['experiment_id'] = experiment_neuron_table['experiment_id'].astype('int64')
    experiment_neuron_table['session_id'] = experiment_neuron_table['session_id'].astype('int64')
    experiment_neuron_table['neuron_id'] = experiment_neuron_table['neuron_id'].astype('int64')
    experiment_neuron_table['quality'] = experiment_neuron_table['quality'].astype('float')
    experiment_neuron_table['neuron_id'] = experiment_neuron_table['neuron_id'] - 1

    filtered_data = filter_and_merge_data(
        experiment_session_table, experiment_neuron_table,
        selected_experiment_ids=[0],
        selected_stimulus_types=[1, 2],
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

    # construct the query array for query encoder
    query_df = pd.DataFrame(query_array, columns=['experiment_id', 'neuron_id'])
    query_array = pd.merge(query_df, experiment_info_table, on='experiment_id', how='left')
    query_array = query_array[['experiment_id', 'species_id', 'sex_id', 'neuron_id']]
    query_array['neuron_unique_id'] = query_array['experiment_id'] * 10000 + query_array['neuron_id']
    query_array = query_array.drop(['neuron_id'], axis=1)
    query_array = query_array.to_numpy()

    # Encode series_ids into query arrays
    max_values = {'Experiment': 1000, 'Species': 9, 'Sex': 3, 'Neuron': 10000000}
    lengths = {'Experiment': 7, 'Species': 2, 'Sex': 1, 'Neuron': 15}
    shuffle_components = ['Neuron']
    query_encoder = SeriesEncoder(max_values, lengths, shuffle_components=shuffle_components)
    query_array = query_encoder.encode(query_array)
    logging.info(f'query_array size:{query_array.shape} \n')

    # Get how many unique cells are there
    train_indices, val_indices = train_val_split(len(data_array), args.chunk_size, test_size=1 - args.train_proportion)
    # get dataset
    train_dataset = RetinalDataset(data_array, query_index, firing_rate_array, image_root_dir, train_indices,
                                   args.chunk_size, device=device, cache_size=args.cache_size,
                                   image_loading_method=args.image_loading_method)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    check_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    dataiter = iter(check_loader)
    movie, labels, index = next(dataiter)
    logging.info(f'movie clip: {movie.shape} labels:{labels} index:{index} \n')
    queryvec = torch.from_numpy(query_array).unsqueeze(1)
    queryvec = queryvec[index]
    logging.info(f'query vector: {queryvec.shape} \n')

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
                                   use_phase_shift=args.use_phase_shift,
                                   use_dense_frequency=args.use_dense_frequency).to(device)
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

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model, optimizer = checkpoint_loader.load_checkpoint(model, optimizer)

if __name__ == "__main__":
    main()