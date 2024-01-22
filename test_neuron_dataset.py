import torch
from datasets.neuron_dataset import RetinalDataset, DataConstructor
from datasets.neuron_dataset import train_val_split, load_mat_to_dataframe, load_data_from_excel, filter_and_merge_data
from torch.utils.data import DataLoader

chunk_size = 50  # Example chunk size
seq_len = 50
stride = 2
image_root_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/'
link_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Link/'
resp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Response/'
exp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/ExperimentSheets.xlsx'
neu_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/experiment_neuron_011724.mat'

experiment_session_table = load_data_from_excel(exp_dir, 'experiment_session')
experiment_session_table = experiment_session_table.drop('stimulus_type', axis=1)

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
# make sure the neuron id is matching while indexing
experiment_neuron_table['neuron_id'] = experiment_neuron_table['neuron_id'] - 1

filtered_data = filter_and_merge_data(
    experiment_session_table, experiment_neuron_table,
    selected_experiment_ids=[0],
    selected_stimulus_types=[1, 2],
    excluded_session_table=None,
    excluded_neuron_table=None
)

# construct the array for dataset
data_constructor = DataConstructor(filtered_data, seq_len=seq_len, stride=stride, link_dir=link_dir, resp_dir=resp_dir)
data_array, query_array, query_index, firing_rate_array = data_constructor.construct_data()
data_array = data_array.astype('int32')
query_array = query_array.astype('int32')
query_index = query_index.astype('int32')
firing_rate_array = firing_rate_array.astype('float32')
# construct the query array for query encoder
query_df = pd.DataFrame(query_array, columns=['experiment_id', 'neuron_id'])
query_array = pd.merge(query_df, experiment_info_table, on='experiment_id', how='left')
query_array = query_array[['experiment_id', 'species_id', 'sex_id', 'neuron_id']]
query_array = query_array.to_numpy()
# get data spit with chucks
train_indices, val_indices = train_val_split(len(data_array), chunk_size)
# get dataset
train_dataset = RetinalDataset(data_array, query_index, firing_rate_array, image_root_dir, train_indices, chunk_size, device='cuda', use_path_cache = False, use_image_cache=True)


check_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
dataiter = iter(check_loader)
movie, labels, index = next(dataiter)

print(f'movie shape: {movie.shape}')
print(f'labels shape: {labels.shape}')
print(f'index shape: {index.shape}')
print(index)