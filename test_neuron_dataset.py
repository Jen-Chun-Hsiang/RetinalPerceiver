import torch
from datasets.neuron_dataset import RetinalDataset
from datasets.neuron_dataset import train_val_split, load_mat_to_dataframe, load_data_from_excel, filter_and_merge_data

chunk_size = 50  # Example chunk size
image_root_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/VideoSpikeDataset/TrainingSet/Stimulus/'

# Splitting into training and validation sets
train_indices, val_indices = train_val_split(len(data_array), chunk_size)

file_path = '/storage1/fs1/KerschensteinerD/Active/Emily/VideoSpikeDataset/ExperimentSheets.xlsx'
experiment_session_table = load_data_from_excel(file_path, 'experiment_session')
experiment_session_table = experiment_session_table.drop('stimulus_type', axis=1)
file_path = '/storage1/fs1/KerschensteinerD/Active/Emily/VideoSpikeDataset/experiment_neuron_011724.mat'
experiment_neuron_table = load_mat_to_dataframe(file_path)


filtered_data = filter_and_merge_data(
    experiment_session_table, experiment_neuron_table,
    quality_threshold=0.3,
    selected_experiment_ids=[0],
    selected_stimulus_types=[1, 2],
    excluded_session_table=None,
    excluded_neuron_table=None
)

constructor = TemporalArrayConstructor(time_id, seq_len, stride)
# (1) use filtered_data and Link to create data_array (exp_id, ses_id, neu_id, frame_id-k, ..., frame_id-1)

# (2) use filtered_data and the ExperimentSheets.xlsx to create quarry array

# use (1) and (2) to create query_series that matches the size of data_array

# Creating datasets
train_dataset = RetinalDataset(data_array, query_series, image_root_dir, train_indices, chunk_size, device='cuda', use_cache=True)
val_dataset = RetinalDataset(data_array, query_series, image_root_dir, val_indices, chunk_size, device='cuda', use_cache=True)
