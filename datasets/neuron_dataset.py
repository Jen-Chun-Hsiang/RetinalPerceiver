import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import random
import pandas as pd
from scipy.io import loadmat
import pandas as pd


def precompute_image_paths(data_array, root_dir):
    precomputed_paths = {}
    for row in data_array:
        experiment_idx = row[0]
        session_idx = row[1]
        image_frame_indices = row[2:]
        for image_frame_idx in image_frame_indices:
            key = (experiment_idx, session_idx, image_frame_idx)
            path = os.path.join(root_dir,
                                f"experiment_{experiment_idx}",
                                f"session_{session_idx}",
                                f"frame_{image_frame_idx}.png")
            precomputed_paths[key] = path
    return precomputed_paths


class RetinalDataset(Dataset):
    def __init__(self, data_array, query_series, image_root_dir, chunk_indices, chunk_size, device='cuda', use_cache=True):
        self.data_array = data_array
        self.query_series = query_series
        self.image_root_dir = image_root_dir
        self.chunk_indices = chunk_indices
        self.chunk_size = chunk_size
        self.device = device
        self.use_cache = use_cache
        self.image_paths = precompute_image_paths(data_array, image_root_dir)
        self.image_tensor_cache = {}
        assert len(data_array) == len(query_series), "data_array and query_series must be the same length"

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        # Determine the range of the chunk
        start_idx = self.chunk_indices[idx]
        end_idx = min(start_idx + self.chunk_size, len(self.data_array))

        # Randomly select a data point within the chunk
        random_idx = random.randint(start_idx, end_idx - 1)
        firing_rate, experiment_id, session_id, neuron_id, *frame_ids = self.data_array[random_idx]
        query_id = self.query_series[random_idx]

        # Load and stack images for the selected data point
        images = [self.load_image(experiment_id, session_id, frame_id) for frame_id in frame_ids]
        images_3d = torch.stack(images, dim=0)

        return images_3d, firing_rate, query_id

    def load_image(self, experiment_id, session_id, frame_id):
        key = (experiment_id, session_id, frame_id)
        if self.use_cache and key in self.image_tensor_cache:
            return self.image_tensor_cache[key]
        image_path = self.image_paths[key]
        image = Image.open(image_path)
        image_tensor = torch.from_numpy(np.array(image)).float().to(self.device)
        if self.use_cache:
            self.image_tensor_cache[key] = image_tensor
        return image_tensor


def train_val_split(data_length, chunk_size, test_size=0.2):
    total_chunks = (data_length - 1) // chunk_size + 1
    indices = np.arange(0, chunk_size * total_chunks, chunk_size)[:total_chunks]
    val_size = int(total_chunks * test_size)
    val_indices = np.random.choice(indices, size=val_size, replace=False)
    train_indices = np.setdiff1d(indices, val_indices)
    return train_indices, val_indices


def load_mat_to_numpy(mat_file_path, variable_name):
    # Load the .mat file
    mat_data = loadmat(mat_file_path)

    # Load the specified variable as a numpy array
    return np.array(mat_data[variable_name])


def load_mat_to_dataframe(mat_file_path, variable_name, column_names):
    # Load the .mat file
    mat_data = loadmat(mat_file_path)

    # Load the specified variable
    array_data = mat_data[variable_name]

    # Convert MATLAB cell array to a list of strings for column names if necessary
    if isinstance(column_names, np.ndarray) and column_names.dtype == object:
        column_names = [str(name[0]) for name in column_names[0]]

    # Create and return a DataFrame
    return pd.DataFrame(array_data, columns=column_names)


def load_data_from_excel(file_path, sheet_name):
    # Load each table from a separate sheet in the Excel file
    table = pd.read_excel(file_path, sheet_name=sheet_name)
    # Create a DataFrame
    df = pd.DataFrame(table)

    return df


def filter_and_merge_data(exp_session_table, exp_neuron_table, quality_threshold, selected_experiment_ids, selected_stimulus_types, excluded_session_table=None, excluded_neuron_table=None):
    # Create compound keys for merging and exclusion
    exp_session_table['experiment_session'] = exp_session_table['experiment_id'].astype(str) + '_' + exp_session_table['session_id'].astype(str)
    exp_neuron_table['experiment_session'] = exp_neuron_table['experiment_id'].astype(str) + '_' + exp_neuron_table['session_id'].astype(str)
    exp_neuron_table['experiment_neuron'] = exp_neuron_table['experiment_id'].astype(str) + '_' + exp_neuron_table['neuron_id'].astype(str)

    # Handle excluded_session_table if provided
    if excluded_session_table is not None:
        excluded_sessions = pd.DataFrame(excluded_session_table)
        excluded_sessions['experiment_session'] = excluded_sessions['experiment_id'].astype(str) + '_' + excluded_sessions['session_id'].astype(str)
    else:
        excluded_sessions = pd.DataFrame(columns=['experiment_session'])

    # Handle excluded_neuron_table if provided
    if excluded_neuron_table is not None:
        excluded_neurons = pd.DataFrame(excluded_neuron_table)
        excluded_neurons['experiment_neuron'] = excluded_neurons['experiment_id'].astype(str) + '_' + excluded_neurons['neuron_id'].astype(str)
    else:
        excluded_neurons = pd.DataFrame(columns=['experiment_neuron'])

    # Merge DataFrames on 'experiment_session'
    merged_df = pd.merge(exp_session_table, exp_neuron_table, on='experiment_session')

    # Apply filters
    if selected_experiment_ids:
        merged_df = merged_df[merged_df['experiment_id'].isin(selected_experiment_ids)]
    if selected_stimulus_types:
        merged_df = merged_df[merged_df['stimulus_type_id'].isin(selected_stimulus_types)]

    # Exclude sessions and neurons
    merged_df = merged_df[~merged_df['experiment_session'].isin(excluded_sessions['experiment_session'])]
    merged_df = merged_df[~merged_df['experiment_neuron'].isin(excluded_neurons['experiment_neuron'])]

    # Filter by response quality
    filtered_df = merged_df[merged_df['response_quality'] >= quality_threshold]

    # Select specific columns to return
    result_df = filtered_df[['neuron_id', 'experiment_id', 'session_id']]

    return result_df


class TemporalArrayConstructor:
    def __init__(self, time_id, seq_len, stride=2, flip_lr=False):
        self.time_id = np.asarray(time_id)
        self.seq_len = seq_len
        self.stride = stride
        self.flip_lr = flip_lr
        self.valid_starts = self._compute_valid_starts()

    def _create_reference_ids(self, time_id):
        """Create reference IDs corresponding to indices in stim_sequence."""
        reference_ids = np.cumsum(time_id) * time_id
        # Adjust to ensure indexing starts from 0 for stim_sequence
        reference_ids -= 1

        return reference_ids

    def _compute_valid_starts(self):
        """Compute valid start indices based on time_id, seq_len, and stride."""
        diff = np.diff(self.time_id, prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        valid_starts = []
        for start, end in zip(starts, ends):
            valid_range = np.arange(start, min(end - self.seq_len + 1, len(self.time_id)), self.stride)
            valid_starts.extend(valid_range)

        return np.array(valid_starts)

    def _construct_array_full_length(self, stim_sequence):
        # Preallocate array for efficiency
        sequences = np.empty((len(self.valid_starts), self.seq_len), dtype=stim_sequence.dtype)

        # Populate the array using vectorized operations
        for idx, start in enumerate(self.valid_starts):
            sequences[idx] = stim_sequence[start:start + self.seq_len][::-1 if self.flip_lr else 1]

        return sequences

    def construct_array(self, stim_sequence):
        if len(stim_sequence) != len(self.time_id):
            reference_ids = self._construct_array_full_length(self._create_reference_ids(self.time_id))
            if np.any(reference_ids == -1):
                raise ValueError("Array contains -1")
            # Constructing the new array
            return np.array([stim_sequence[index] for index in reference_ids])
        else:
            return self._construct_array_full_length(stim_sequence)


class DataConstructor:
    def __init__(self, input_table, seq_len, stride, link_dir, resp_dir):
        self.input_table = input_table
        self.seq_len = seq_len
        self.stride = stride
        self.link_dir = link_dir
        self.resp_dir = resp_dir

    def construct_data(self):
        all_sessions_data = []
        grouped = self.input_table.groupby(['experiment_id', 'session_id'])

        for (experiment_id, session_id), group in grouped:
            neurons = group['neuron_id'].unique()

            file_path = os.path.join(self.link_dir, f'experiment_{experiment_id}/session_{session_id}.mat')
            time_id = load_mat_to_numpy(file_path, 'time_id')
            video_frame_id = load_mat_to_numpy(file_path, 'video_frame_id')

            sequence_id = np.arange(len(video_frame_id))

            constructor = TemporalArrayConstructor(time_id=time_id, seq_len=self.seq_len, stride=self.stride)
            session_array = constructor.construct_array(video_frame_id)

            file_path = os.path.join(self.resp_dir, f'experiment_{experiment_id}/session_{session_id}.mat')
            firing_rate_array = load_mat_to_numpy(file_path, 'spike_smooth')

            firing_rate_index = construct_temporal_array_imageid(sequence_id, self.seq_len, self.stride, flip_lr=True)

            num_rows = len(session_array) * len(neurons)
            session_data = np.empty((num_rows, 4 + self.seq_len - 1), dtype=session_array.dtype)

            for i, neuron_id in enumerate(neurons):
                start_row = i * len(session_array)
                end_row = start_row + len(session_array)

                session_data[start_row:end_row, :3] = [experiment_id, session_id, neuron_id]

                # Firing rate data as the fourth column
                firing_rate_data = firing_rate_array[firing_rate_index[:, 0], neuron_id]
                session_data[start_row:end_row, 3] = firing_rate_data

                # Remaining columns filled with session_array
                session_data[start_row:end_row, 4:] = session_array

            all_sessions_data.append(session_data)

        final_data = np.vstack(all_sessions_data)
        return final_data


def construct_temporal_array_imageid(stim_sequence, seq_len, stride=2, flip_lr=False):
    resp_seq_len = (len(stim_sequence) - seq_len) // stride + 1
    imgids = np.zeros((resp_seq_len, seq_len), dtype=int)

    for i in range(resp_seq_len):
        start_idx = i * stride
        sequence_slice = stim_sequence[start_idx:start_idx + seq_len]
        imgids[i, :] = sequence_slice[::-1] if flip_lr else sequence_slice

    return imgids



