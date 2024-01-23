import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import random
import pandas as pd
from scipy.io import loadmat


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
    def __init__(self, data_array, query_series, firing_rate_array, image_root_dir, chunk_indices, chunk_size, device='cuda', use_path_cache=True, use_image_cache=True):
        self.data_array = data_array
        self.query_series = query_series
        self.firing_rate_array = firing_rate_array
        self.image_root_dir = image_root_dir
        self.chunk_indices = chunk_indices
        self.chunk_size = chunk_size
        self.device = device
        self.use_path_cache = use_path_cache
        self.use_image_cache = use_image_cache

        if self.use_path_cache:
            self.image_paths = precompute_image_paths(data_array, image_root_dir)
        else:
            self.image_paths = None

        if self.use_image_cache:
            self.image_tensor_cache = {}
        else:
            self.image_tensor_cache = None

        assert len(data_array) == len(query_series), "data_array and query_series must be the same length"

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        # Determine the range of the chunk
        start_idx = self.chunk_indices[idx]
        end_idx = min(start_idx + self.chunk_size, len(self.data_array))

        # Randomly select a data point within the chunk
        random_idx = random.randint(start_idx, end_idx - 1)
        experiment_id, session_id, neuron_id, *frame_ids = self.data_array[random_idx]
        firing_rate = self.firing_rate_array[random_idx]
        query_id = self.query_series[random_idx]

        # Load and stack images for the selected data point
        images = [self.load_image(experiment_id, session_id, frame_id) for frame_id in frame_ids]
        images_3d = torch.stack(images, dim=0)
        images_3d = images_3d.unsqueeze(0)

        return images_3d, firing_rate, query_id

    def load_image(self, experiment_id, session_id, frame_id):
        key = (experiment_id, session_id, frame_id)

        if self.use_image_cache and key in self.image_tensor_cache:
            return self.image_tensor_cache[key]

        if self.use_path_cache:
            image_path = self.image_paths[key]
        else:
            image_path = os.path.join(self.image_root_dir,
                                      f"experiment_{experiment_id}",
                                      f"session_{session_id}",
                                      f"frame_{frame_id}.png")

        image = Image.open(image_path)
        image_tensor = torch.from_numpy(np.array(image)).float().to(self.device)
        image_tensor = (image_tensor / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]

        if self.use_image_cache:
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


def load_mat_to_dataframe(mat_file_path, data_variable_name, column_names_variable):
    # Load the .mat file
    mat_data = loadmat(mat_file_path)

    # Load the data using the specified variable name
    array_data = mat_data[data_variable_name]

    # Load the column names using the specified variable name
    column_names_array = mat_data[column_names_variable]

    # Convert MATLAB cell array to a list of strings for column names if necessary
    if isinstance(column_names_array, np.ndarray) and column_names_array.dtype == object:
        column_names = [str(name[0]) for name in column_names_array.flatten()]
    else:
        raise ValueError("Column names should be provided in a MATLAB cell array format.")

    # Create and return a DataFrame
    return pd.DataFrame(array_data, columns=column_names)


def load_data_from_excel(file_path, sheet_name):
    # Load each table from a separate sheet in the Excel file
    table = pd.read_excel(file_path, sheet_name=sheet_name)
    # Create a DataFrame
    df = pd.DataFrame(table)

    return df


def filter_and_merge_data(exp_session_table, exp_neuron_table, selected_experiment_ids, selected_stimulus_types, excluded_session_table=None, excluded_neuron_table=None):
    # Filter exp_session_table before merging
    if selected_experiment_ids:
        exp_session_table = exp_session_table[exp_session_table['experiment_id'].isin(selected_experiment_ids)]
    if selected_stimulus_types:
        exp_session_table = exp_session_table[exp_session_table['stimulus_type_id'].isin(selected_stimulus_types)]

    # Merge DataFrames on 'experiment_id' and 'session_id'
    merged_df = pd.merge(exp_session_table, exp_neuron_table, on=['experiment_id', 'session_id'])

    # Handle excluded_session_table if provided
    if excluded_session_table is not None:
        excluded_sessions = excluded_session_table.copy()
        excluded_sessions['exclude'] = True
        merged_df = pd.merge(merged_df, excluded_sessions[['experiment_id', 'session_id', 'exclude']], on=['experiment_id', 'session_id'], how='left')
        merged_df = merged_df[merged_df['exclude'] != True]
        merged_df.drop(columns=['exclude'], inplace=True)

    # Handle excluded_neuron_table if provided
    if excluded_neuron_table is not None:
        excluded_neurons = excluded_neuron_table.copy()
        excluded_neurons['exclude'] = True
        merged_df = pd.merge(merged_df, excluded_neurons[['experiment_id', 'neuron_id', 'exclude']], on=['experiment_id', 'neuron_id'], how='left')
        merged_df = merged_df[merged_df['exclude'] != True]
        merged_df.drop(columns=['exclude'], inplace=True)

    # Filter by response quality using quality_threshold from exp_session_table
    filtered_df = merged_df[merged_df['quality'] >= merged_df['quality_threshold']]

    # Select specific columns to return
    result_df = filtered_df[['experiment_id', 'session_id', 'neuron_id']]

    return result_df


class TemporalArrayConstructor:
    def __init__(self, time_id, seq_len, stride=2):
        self.time_id = np.asarray(time_id).ravel()
        self.seq_len = seq_len
        self.stride = stride
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

    def _construct_array_full_length(self, stim_sequence, flip_lr):
        # Preallocate array for efficiency
        sequences = np.empty((len(self.valid_starts), self.seq_len), dtype=stim_sequence.dtype)

        # Populate the array using vectorized operations
        for idx, start in enumerate(self.valid_starts):
            sequences[idx] = stim_sequence[start:start + self.seq_len][::-1 if flip_lr else 1]

        return sequences

    def construct_array(self, stim_sequence, flip_lr=False):
        stim_sequence = stim_sequence.ravel()
        if len(stim_sequence) != len(self.time_id):
            reference_ids = self._construct_array_full_length(self._create_reference_ids(self.time_id), flip_lr)
            if np.any(reference_ids == -1):
                raise ValueError("Array contains -1")
            # Constructing the new array
            return np.array([stim_sequence[index] for index in reference_ids])
        else:
            return self._construct_array_full_length(stim_sequence, flip_lr)


class DataConstructor:
    def __init__(self, input_table, seq_len, stride, link_dir, resp_dir):
        self.input_table = input_table
        self.seq_len = seq_len
        self.stride = stride
        self.link_dir = link_dir
        self.resp_dir = resp_dir

    def construct_data(self):
        all_sessions_data = []
        all_sessions_fr_data = []
        grouped = self.input_table.groupby(['experiment_id', 'session_id'])

        for (experiment_id, session_id), group in grouped:
            neurons = group['neuron_id'].unique()
            file_path = os.path.join(self.link_dir, f'experiment_{experiment_id}/session_{session_id}.mat')
            time_id = load_mat_to_numpy(file_path, 'time_id')
            video_frame_id = load_mat_to_numpy(file_path, 'video_frame_id')

            sequence_id = np.arange(len(video_frame_id))

            constructor = TemporalArrayConstructor(time_id=time_id, seq_len=self.seq_len, stride=self.stride)
            session_array = constructor.construct_array(video_frame_id)
            session_array = session_array.astype(np.int32)

            file_path = os.path.join(self.resp_dir, f'experiment_{experiment_id}/session_{session_id}.mat')
            firing_rate_array = load_mat_to_numpy(file_path, 'spike_smooth')

            firing_rate_index = constructor.construct_array(sequence_id, flip_lr=True)
            num_rows = len(session_array) * len(neurons)
            session_data = np.empty((num_rows, 3 + self.seq_len), dtype=session_array.dtype)
            session_fr_data = np.empty((num_rows, 1))
            for i, neuron_id in enumerate(neurons):
                start_row = i * len(session_array)
                end_row = start_row + len(session_array)

                session_data[start_row:end_row, :3] = [experiment_id, session_id, neuron_id]

                # Firing rate data as the fourth column
                firing_rate_data = firing_rate_array[firing_rate_index[:, 0], neuron_id]

                session_fr_data[start_row:end_row, 0] = firing_rate_data

                # Remaining columns filled with session_array
                session_data[start_row:end_row, 3:] = session_array

            all_sessions_data.append(session_data)
            all_sessions_fr_data.append(session_fr_data)

        frame_array = np.vstack(all_sessions_data)
        firing_rate_array = np.vstack(all_sessions_fr_data)

        query_array, query_index = np.unique(frame_array[:, [0, 2]], axis=0, return_inverse=True)

        return frame_array, query_array, query_index, firing_rate_array




