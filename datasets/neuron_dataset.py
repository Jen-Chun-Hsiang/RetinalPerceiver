import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from PIL import Image
import os
import random
import pandas as pd
from scipy.io import loadmat
from collections import OrderedDict
import threading
import h5py

# from functools import lru_cache
# from torchvision.io import read_image
# from torchvision.transforms.functional import convert_image_dtype



def get_frames_by_indices(hdf5_file, frame_indices, image_size=(800, 600)):
    """
    Retrieve specific frames from an HDF5 file based on a set of indices with pre-allocation.

    :param hdf5_file: Path to the HDF5 file.
    :param frame_indices: List of frame indices to retrieve.
    :param image_size: Tuple indicating the size of the images (height, width).
    :return: List of PyTorch tensors representing the requested frames.
    """
    img_height, img_width = image_size
    num_channels = 3  # Assuming RGB images; change if different

    # Pre-allocate a list for the frames
    frames = [torch.empty((num_channels, img_height, img_width), dtype=torch.float32) for _ in frame_indices]

    with h5py.File(hdf5_file, 'r') as hfile:
        for i, index in enumerate(frame_indices):
            str_index = str(index)
            if str_index in hfile:
                img_array = np.array(hfile[str_index])
                frames[i] = torch.tensor(img_array, dtype=torch.float32)
            else:
                print(f"Frame index {index} not found in HDF5 file.")

    return frames


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
    def __init__(self, data_array, query_series, firing_rate_array, root_dir, chunk_indices, chunk_size,
                 device='cuda', cache_size=100, image_loading_method='hdf5'):
        """
        Initializes the RetinalDataset.

        Parameters:
        - data_array: Information about each data point (experiment ID, session ID, neuron ID, frame IDs).
        - query_series: Array containing query IDs.
        - firing_rate_array: Array containing the firing rates corresponding to each data point.
        - image_root_dir: Root directory where images are stored.
        - chunk_indices: Indices to define chunks of data for processing.
        - chunk_size: Size of each chunk.
        - device: The device (CPU/GPU) where tensors will be allocated.
        - cache_size: The maximum number of images to store in the cache.
        """
        self.data_array = data_array
        self.query_series = query_series
        self.firing_rate_array = firing_rate_array
        self.root_dir = root_dir
        self.chunk_indices = chunk_indices
        self.chunk_size = chunk_size
        self.device = device
        self.cache_size = cache_size
        self.image_loading_method = image_loading_method
        self.image_tensor_cache = OrderedDict()
        self.cache_lock = threading.Lock()  # Create a lock for the cache

        assert len(data_array) == len(query_series), "data_array and query_series must be the same length"

        if image_loading_method == 'hdf5':
            example_session_path = self.get_hdf5_path(*self.data_array[0][:2])
            with h5py.File(example_session_path, 'r') as hfile:
                sample_frame_id = self.data_array[0][3]
                sample_image_tensor = torch.tensor(np.array(hfile[str(sample_frame_id)])).float().to(self.device)
                self.image_shape = sample_image_tensor.shape
        else:
            # Dummy load to get image size for 'png' or 'pt' formats
            experiment_id, session_id, neuron_id = self.data_array[0][:3]
            frame_id = self.data_array[0][3]
            sample_image_tensor = self.load_image(experiment_id, session_id, frame_id)
            self.image_shape = sample_image_tensor.shape

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        """
         Retrieves a chunk of data by index.

         Parameters:
         - idx: Index of the chunk.

         Returns:
         - images_3d: A tensor containing stacked images for the selected data point.
         - firing_rate: The firing rate associated with the selected data point.
         - query_id: The query ID associated with the selected data point.
         """
        # Determine the range of the chunk
        start_idx = self.chunk_indices[idx]
        end_idx = min(start_idx + self.chunk_size, len(self.data_array))

        # Randomly select a data point within the chunk
        random_idx = random.randint(start_idx, end_idx - 1)
        experiment_id, session_id, neuron_id, *frame_ids = self.data_array[random_idx]
        firing_rate = self.firing_rate_array[random_idx]
        query_id = self.query_series[random_idx]

        # Find unique frame_ids and their indices
        unique_frame_ids, inverse_indices = np.unique(frame_ids, return_inverse=True)

        # Initialize an empty tensor to hold all images
        images_3d = torch.empty((len(frame_ids),) + self.image_shape[-2:], device=self.device)

        if self.image_loading_method == 'hdf5':
            hdf5_path = self.get_hdf5_path(experiment_id, session_id)
            unique_frames = self.get_frames_by_indices(hdf5_path, unique_frame_ids)

            # Assign the loaded images to the respective positions in images_3d
            for unique_idx, image in zip(np.unique(inverse_indices), unique_frames):
                indices = np.where(inverse_indices == unique_idx)[0]
                for i in indices:
                    images_3d[i] = image
        else:

            # For 'png' or 'pt', load each unique image and assign to images_3d
            for final_idx, frame_id in enumerate(frame_ids):
                image = self.load_image(experiment_id, session_id, frame_id)
                images_3d[final_idx] = image
            '''
            for unique_idx in np.unique(inverse_indices):
                frame_id = unique_frame_ids[unique_idx]
                image = self.load_image(experiment_id, session_id, frame_id)
                indices = np.where(inverse_indices == unique_idx)[0]
                for i in indices:
                    images_3d[i] = image
            '''

        images_3d = images_3d.unsqueeze(0).to(self.device)  # Adding an extra dimension to simulate batch size

        return images_3d, firing_rate, query_id

    def get_hdf5_path(self, experiment_id, session_id):
        """Construct the path to the HDF5 file for a given experiment and session."""
        return os.path.join(self.root_dir, f"experiment_{experiment_id}", f"session_{session_id}.hdf5")

    def get_frames_by_indices(self, hdf5_file, frame_indices):
        """
        Retrieve specific frames from an HDF5 file based on a set of indices.
        """
        img_height, img_width = self.image_shape[0], self.image_shape[1]

        # Pre-allocate a list for the frames
        frames = [torch.empty((img_height, img_width), dtype=torch.float32) for _ in frame_indices]

        with h5py.File(hdf5_file, 'r') as hfile:
            for i, index in enumerate(frame_indices):
                str_index = str(index)
                if str_index in hfile:
                    img_array = np.array(hfile[str_index])
                    frames[i] = torch.tensor(img_array, dtype=torch.float32)
                else:
                    print(f"Frame index {index} not found in HDF5 file.")
        return frames

    def load_image(self, experiment_id, session_id, frame_id):
        key = (experiment_id, session_id, frame_id)

        # Synchronized access to the cache
        with self.cache_lock:
            # Check if the image is in the cache
            if key in self.image_tensor_cache:
                # Move the item to the end of the cache to mark it as recently used
                self.image_tensor_cache.move_to_end(key)
                return self.image_tensor_cache[key]

        image_tensor = None
        if self.image_loading_method == 'png':
            image_path = self.get_image_path(experiment_id, session_id, frame_id, '.png')
            image = Image.open(image_path)
            image_tensor = torch.from_numpy(np.array(image)).float().to(self.device)
            image_tensor = (image_tensor / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]
        elif self.image_loading_method == 'pt':
            image_path = self.get_image_path(experiment_id, session_id, frame_id, '.pt')
            image_tensor = torch.load(image_path, map_location=self.device)

        # Synchronized cache update
        with self.cache_lock:
            # Add to cache and enforce cache size limit
            self.image_tensor_cache[key] = image_tensor
            if len(self.image_tensor_cache) > self.cache_size:
                self.image_tensor_cache.popitem(last=False)  # Remove the oldest item

        return image_tensor

    def get_image_path(self, experiment_id, session_id, frame_id, extension):
        return os.path.join(self.root_dir,
                            f"experiment_{experiment_id}",
                            f"session_{session_id}",
                            f"frame_{frame_id}{extension}")


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


def filter_and_merge_data(exp_session_table, exp_neuron_table, selected_experiment_ids, selected_stimulus_types,
                          excluded_session_table=None, excluded_neuron_table=None,
                          included_session_table=None, included_neuron_table=None):
    # Filter exp_session_table before merging
    if selected_experiment_ids:
        exp_session_table = exp_session_table[exp_session_table['experiment_id'].isin(selected_experiment_ids)]
    if selected_stimulus_types:
        exp_session_table = exp_session_table[exp_session_table['stimulus_type_id'].isin(selected_stimulus_types)]

    # Include sessions if included_session_table is provided
    if included_session_table is not None:
        included_sessions = included_session_table[['experiment_id', 'session_id']]
        exp_session_table = pd.merge(exp_session_table, included_sessions, on=['experiment_id', 'session_id'], how='inner')

    # Include neurons if included_neuron_table is provided before merging
    if included_neuron_table is not None:
        included_neurons = included_neuron_table[['experiment_id', 'neuron_id']]
        exp_neuron_table = pd.merge(exp_neuron_table, included_neurons, on=['experiment_id', 'neuron_id'], how='inner')

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

                # Firing rate data as the fourth column (Get the id correct by -1)
                firing_rate_data = firing_rate_array[firing_rate_index[:, 0], neuron_id-1]

                session_fr_data[start_row:end_row, 0] = firing_rate_data

                # Remaining columns filled with session_array
                session_data[start_row:end_row, 3:] = session_array

            all_sessions_data.append(session_data)
            all_sessions_fr_data.append(session_fr_data)

        frame_array = np.vstack(all_sessions_data)
        firing_rate_array = np.vstack(all_sessions_fr_data)

        assert len(frame_array) == len(firing_rate_array)

        query_array, query_index = np.unique(frame_array[:, [0, 2]], axis=0, return_inverse=True)

        return frame_array, query_array, query_index, firing_rate_array


class GroupedSampler(Sampler):
    def __init__(self, data_source, neuron_links, batch_size):
        self.data_source = data_source
        self.neuron_links = neuron_links
        self.batch_size = batch_size
        self._group_indices()

    def _group_indices(self):
        # Compute unique links and group indices
        unique_links = np.unique(self.neuron_links)
        self.grouped_indices = {link: np.where(self.neuron_links == link)[0].tolist() for link in unique_links}
        self._prepare_batches()

    def _prepare_batches(self):
        # Prepare batches from grouped indices without shuffling each group individually
        self.batches = []
        for group in self.grouped_indices.values():
            for i in range(0, len(group), self.batch_size):
                self.batches.append(group[i:i + self.batch_size])

        # Shuffle the batches after all have been created
        random.shuffle(self.batches)

    def reshuffle(self):
        # Call this method to reshuffle the batches at the beginning of each epoch
        self._prepare_batches()

    def __iter__(self):
        # Optionally reshuffle here if you want automatic reshuffling every epoch
        # self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)



