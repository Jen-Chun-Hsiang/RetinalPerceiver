import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from datasets.examine_dataset import STA_check
import torch
from datetime import datetime
from datasets.neuron_dataset import RetinalDataset, DataConstructor
from datasets.neuron_dataset import train_val_split, load_mat_to_dataframe, load_data_from_excel, filter_and_merge_data
from utils.utils import DataVisualizer

def main():
    batch_size = 16
    chunk_size = 50
    input_depth = 50
    data_stride = 2
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    image_root_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/'
    link_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Link/'
    resp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Response/'
    exp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/ExperimentSheets.xlsx'
    neu_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/experiment_neuron_021324.mat'

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    experiment_session_table = load_data_from_excel(exp_dir, 'experiment_session')
    experiment_session_table = experiment_session_table.drop('stimulus_type', axis=1)

    included_neuron_table = load_data_from_excel(exp_dir, 'included_neuron_03')

    experiment_info_table = load_data_from_excel(exp_dir, 'experiment_info')
    experiment_info_table = experiment_info_table.drop(['species', 'sex', 'day', 'folder_name'], axis=1)

    experiment_neuron_table = load_mat_to_dataframe(neu_dir, 'experiment_neuron_table', 'column_name')
    experiment_neuron_table.iloc[:, 0:3] = experiment_neuron_table.iloc[:, 0:3].astype('int64')
    # make sure the format is correct
    experiment_neuron_table.fillna(-1, inplace=True)
    experiment_neuron_table['experiment_id'] = experiment_neuron_table['experiment_id'].astype('int64')
    experiment_neuron_table['session_id'] = experiment_neuron_table['session_id'].astype('int64')
    experiment_neuron_table['neuron_id'] = experiment_neuron_table['neuron_id'].astype('int64')
    experiment_neuron_table['quality'] = experiment_neuron_table['quality'].astype('float')

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
    data_constructor = DataConstructor(filtered_data, seq_len=input_depth, stride=data_stride,
                                       link_dir=link_dir, resp_dir=resp_dir)
    data_array, query_array, query_index, firing_rate_array = data_constructor.construct_data()
    data_array = data_array.astype('int64')
    query_array = query_array.astype('int64')
    query_index = query_index.astype('int64')
    firing_rate_array = firing_rate_array.astype('float32')

    print(f'data_array shape: {data_array.shape}')
    train_indices, val_indices = train_val_split(len(data_array), chunk_size, test_size=1 - 0.2)
    train_dataset = RetinalDataset(data_array, query_index, firing_rate_array, image_root_dir, train_indices,
                                   chunk_size, device=device, cache_size=80,
                                   image_loading_method='pt')
    output_image = STA_check(train_dataset, batch_size=batch_size, device=device)

    # Initialize the DataVisualizer
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    visualizer_est_rf = DataVisualizer(savefig_dir, file_prefix=f'current_Estimate_RF_{timestr}')

    output_image_np = output_image.squeeze().cpu().numpy()
    visualizer_est_rf.plot_and_save(output_image_np, plot_type='3D_matrix', num_cols=5)

if __name__ == "__main__":
    main()

