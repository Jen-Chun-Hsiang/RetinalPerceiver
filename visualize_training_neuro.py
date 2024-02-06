import os
from datetime import datetime
import logging
import torch
import pandas as pd

from utils.training_procedure import CheckpointLoader, forward_model
from datasets.neuron_dataset import RetinalDataset, DataConstructor
from datasets.neuron_dataset import train_val_split, load_mat_to_dataframe, load_data_from_excel, filter_and_merge_data
from utils.utils import DataVisualizer, SeriesEncoder
# (0) identify the cell we have modeled their responses
# (1) show the receptive field with the white noise
# (2) show the response to the test set


def main():
    stimulus_type = '50tpcnn_2024020502_GoodCell'  # get the name from the check point folder
    epoch_end = 100  # the number of epoch in the check_point file
    is_full_figure_draw = True  # determine whether draw for each neuro or just get stats
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    checkpoint_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    exp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/ExperimentSheets.xlsx'
    neu_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/experiment_neuron_011724.mat'
    link_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Link/'

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
        included_session_table = None,
        included_neuron_table = included_neuron_table
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

if __name__ == "__main__":
    main()