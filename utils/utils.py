import torch
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# Function to evaluate the model on validation data
def validate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to track gradients during validation
        for input_matrices, targets in dataloader:
            targets = targets.view(targets.size(0), -1)
            outputs = model(input_matrices)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def create_checkpoint_filename(checkpoint_dir, file_prefix='model_checkpoint'):
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d')

    # Check for existing files and determine the next series number
    existing_files = os.listdir(checkpoint_dir)
    max_series_num = 0
    for file in existing_files:
        if file.startswith(file_prefix) and timestamp in file:
            parts = file.split('_')
            try:
                series_num = int(parts[-1].split('.')[0])  # Extract series number
                max_series_num = max(max_series_num, series_num)
            except ValueError:
                # Handle the case where the filename format is unexpected
                continue

    next_series_num = max_series_num + 1

    # Create new checkpoint filename
    new_filename = f"{file_prefix}_{timestamp}_{next_series_num}.pth"
    return new_filename


def plot_and_save_3d_matrix_with_timestamp(target_matrix, num_cols, result_dir, file_prefix='plot_figure'):
    """
    Plots and saves a figure for a 3D matrix with a timestamp and series number in the filename.

    Args:
    target_matrix (numpy.ndarray): A 3D numpy array to be plotted.
    num_cols (int): Number of columns in the subplot grid.
    result_dir (str): Directory where the figure will be saved.
    file_prefix (str): Prefix for the saved figure file name.

    """
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(target_matrix, torch.Tensor):
        # Ensure the tensor is on CPU and is a float32 tensor
        target_matrix = target_matrix.cpu().numpy()

    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Check for existing files and determine the next series number
    existing_files = os.listdir(result_dir)
    max_series_num = 0
    for file in existing_files:
        if file.startswith(file_prefix) and timestamp in file:
            parts = file.split('_')
            try:
                series_num = int(parts[-1].split('.')[0])  # Extract series number
                max_series_num = max(max_series_num, series_num)
            except ValueError:
                # Handle the case where the filename format is unexpected
                continue

    next_series_num = max_series_num + 1

    # Plot the matrix
    num_rows = target_matrix.shape[0] // num_cols + int(target_matrix.shape[0] % num_cols > 0)
    plt.figure(figsize=(15, num_rows * 3))

    for i in range(target_matrix.shape[0]):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        image = ax.imshow(target_matrix[i], cmap='viridis', vmin=-0.4, vmax=0.5)
        plt.title(f'Time Frame {i}')
        plt.axis('off')
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Ensure the result directory exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Create new filename with timestamp and series number
    new_filename = f"{file_prefix}_{timestamp}_{next_series_num}.png"
    file_path = os.path.join(result_dir, new_filename)

    # Save the figure
    plt.savefig(file_path)
    plt.close()  # Close the figure to free up memory

    return file_path


class DataVisualizer:
    def __init__(self, result_dir, file_prefix='plot_figure'):
        """
        Initialize the DataVisualizer with a result directory and a file prefix.

        Args:
        result_dir (str): Directory where the figure will be saved.
        file_prefix (str): Prefix for the saved figure file name.
        """
        self.result_dir = result_dir
        self.file_prefix = file_prefix

    def plot_and_save(self, data, plot_type='3D_matrix', num_cols=3, custom_plot_func=None, **kwargs):
        """
        Plots and saves a figure based on the data and plot type.

        Args:
        data (numpy.ndarray or torch.Tensor): Data to be plotted.
        plot_type (str): Type of the plot ('3D_matrix', 'line', etc.). Use 'custom' for custom_plot_func.
        num_cols (int): Number of columns in the subplot grid for 3D matrix plotting.
        custom_plot_func (function): A custom function for plotting if plot_type is 'custom'.
        """
        # Convert PyTorch tensor to NumPy array if necessary
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # Determine next series number
        next_series_num = self._get_next_series_num(timestamp)

        # Plot the data
        if plot_type == '3D_matrix':
            self._plot_3d_matrix(data, num_cols)
        elif plot_type == '2D_matrix':
            self._plot_2d_matrix(data)
        elif plot_type == 'scatter':
            self._plot_scatter(**kwargs)
        elif plot_type == 'line':
            self._plot_line(**kwargs)
        elif plot_type == 'custom' and custom_plot_func is not None:
            custom_plot_func(data)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        # Save the figure
        file_path = self._save_figure(timestamp, next_series_num)
        return file_path

    def _get_next_series_num(self, timestamp):
        existing_files = os.listdir(self.result_dir)
        max_series_num = 0
        for file in existing_files:
            if file.startswith(self.file_prefix) and timestamp in file:
                parts = file.split('_')
                try:
                    series_num = int(parts[-1].split('.')[0])
                    max_series_num = max(max_series_num, series_num)
                except ValueError:
                    continue
        return max_series_num + 1

    def _save_figure(self, timestamp, series_num):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        new_filename = f"{self.file_prefix}_{timestamp}_{series_num}.png"
        file_path = os.path.join(self.result_dir, new_filename)
        plt.savefig(file_path)
        plt.close()
        return file_path
    def _plot_2d_matrix(self, data):
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='viridis', interpolation='nearest')
        plt.title('2D matrix')
        plt.colorbar()

    def _plot_3d_matrix(self, data, num_cols):
        num_rows = data.shape[0] // num_cols + int(data.shape[0] % num_cols > 0)
        # Calculate the global min and max values across all data samples
        data_min = np.min(data)
        data_max = np.max(data)

        plt.figure(figsize=(15, num_rows * 3))
        for i in range(data.shape[0]):
            ax = plt.subplot(num_rows, num_cols, i + 1)
            image = ax.imshow(data[i], cmap='viridis', vmin=data_min, vmax=data_max)
            plt.title(f'Time Frame {i}')
            plt.axis('off')
            plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

    def _plot_scatter(self, x_data, y_data, xlabel='X-axis', ylabel='Y-axis', title='Scatter Plot'):
        """
        Plots a scatter plot of the given data.

        Args:
        x_data (array-like): Data for the X-axis.
        y_data (array-like): Data for the Y-axis.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str): Title of the plot.
        """
        plt.figure()
        plt.scatter(x_data, y_data, s=5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)

    def _plot_line(self, line1, line2, setlim=[0, 0.001], xlabel='Epochs', ylabel='Losses', title='Training and Validation Loss'):
        """
        Plots a scatter plot of the given data.

        Args:
        x_data (array-like): Data for the X-axis.
        y_data (array-like): Data for the Y-axis.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str): Title of the plot.
        """

        plt.figure()
        plt.plot(range(1, len(line1) + 1), line1, label='Training Loss')
        plt.plot(range(1, len(line2) + 1), line2, label='Validation Loss')
        plt.ylim(setlim)
        plt.legend()
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)


class SeriesEncoder:
    def __init__(self, max_values, lengths, order=None, shuffle_components=None, seed=42):
        """
        Initialize the encoder with maximum values, lengths, and optional order and shuffle settings.
        max_values: Dictionary with keys specifying the maximum values for each component.
        lengths: Dictionary with keys specifying the length of each component.
        order: List specifying the order of the components, default is the order of max_values keys.
        shuffle_components: List of components to be shuffled, default is empty.
        seed: Random seed for consistency.
        """
        self.max_values = max_values
        self.lengths = lengths
        self.order = order if order is not None else list(max_values.keys())
        self.shuffle_components = shuffle_components if shuffle_components is not None else []
        np.random.seed(seed)
        self.bases = self.calculate_bases(max_values, lengths)
        self.shuffle_indices = {component: np.random.permutation(lengths[component]) for component in self.shuffle_components}

    def calculate_bases(self, max_values, lengths):
        """ Calculate optimal bases for each component. """
        bases = {}
        for component, max_val in max_values.items():
            length = lengths[component]
            base = np.ceil((max_val + 1)  ** (1 / length))
            bases[component] = base
        return bases

    def encode_component(self, value, max_value, length, base, component):
        """ Encode a value using base representation and apply shuffling if needed. """
        encoded = np.zeros(length)
        for i in range(length):
            digit = (value % base) / (base - 1) * 2 - 1
            encoded[i] = digit
            value //= base

        if component in self.shuffle_components:
            # Apply shuffling
            shuffle_idx = self.shuffle_indices[component]
            encoded = encoded[shuffle_idx]

        return encoded

    def encode(self, input_tuples):
        """
        Encode multiple input tuples into a concatenated vector.
        input_tuples: List of tuples, each representing values for components in the order.
        """
        encoded_vectors = []

        for input_values in input_tuples:
            encoded_vector = []
            for component, value in zip(self.order, input_values):
                max_value = self.max_values[component]
                length = self.lengths[component]
                base = self.bases[component]
                # randomize = component in self.shuffle_components
                encoded_vector.extend(self.encode_component(value, max_value, length, base, component))
            encoded_vectors.append(np.array(encoded_vector))

        # Concatenate along a new dimension
        return np.stack(encoded_vectors)


def add_gradient(tensor, dim, start=-1, end=1):
    """
    Adds a gradient along a specified dimension of the tensor.

    :param tensor: Input tensor.
    :param dim: Dimension along which the gradient is added.
    :param start: Start value of the gradient (default -1).
    :param end: End value of the gradient (default 1).
    :return: Tensor with added gradient.
    """
    # Number of steps along the dimension
    num_steps = tensor.shape[dim]

    # Create a gradient vector
    gradient = torch.linspace(start, end, steps=num_steps, device=tensor.device, dtype=tensor.dtype)

    # Reshape the gradient to be broadcastable to the tensor shape
    shape = [1] * len(tensor.shape)
    shape[dim] = num_steps
    gradient = gradient.view(shape)

    # Add the gradient to the tensor
    return tensor + gradient


class VideoPatcher:
    def __init__(self, video_shape, kernel_size=(2, 2, 2), stride=(1, 1, 1)):
        """
        Initialize the VideoPatcher with video shape, kernel size, and stride.

        Parameters:
        video_shape (tuple): Shape of the video tensor [C, T, H, W].
        kernel_size (tuple): Size of each patch (T, H, W).
        stride (tuple): Stride in each dimension (T, H, W).
        """
        self.video_shape = video_shape
        self.kernel_size = kernel_size
        self.stride = stride

        # Pre-calculate padding and the number of patches in each dimension
        _, _, T, H, W = video_shape
        self.pad_T = (-(T - kernel_size[0]) % stride[0])
        self.pad_H = (-(H - kernel_size[1]) % stride[1])
        self.pad_W = (-(W - kernel_size[2]) % stride[2])

        self.t_patches = (T + self.pad_T - kernel_size[0]) // stride[0] + 1
        self.h_patches = (H + self.pad_H - kernel_size[1]) // stride[1] + 1
        self.w_patches = (W + self.pad_W - kernel_size[2]) // stride[2] + 1

    def video_to_patches(self, videos):
        """
        Convert a batch of videos to overlapping 3D patches with automatic padding if necessary.
        Output shape: [B, C, number of patches, flatten patch].

        Parameters:
        videos (tensor): Batch of video tensors of shape [B, C, T, H, W].

        Returns:
        patches (tensor): Tensor of patches in the specified shape for each video in the batch.
        """
        B, C, T, H, W = videos.shape

        # Pad the videos if necessary
        if self.pad_T > 0 or self.pad_H > 0 or self.pad_W > 0:
            videos = torch.nn.functional.pad(videos, (0, self.pad_W, 0, self.pad_H, 0, self.pad_T))

        # Unfold the tensor along the temporal, height, and width dimensions
        patches = videos.unfold(2, self.kernel_size[0], self.stride[0]) \
            .unfold(3, self.kernel_size[1], self.stride[1]) \
            .unfold(4, self.kernel_size[2], self.stride[2])

        # Reshape and flatten each patch
        patches = patches.contiguous()
        patches = patches.view(B, C, -1, self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])

        return patches

    def generate_patch_indices(self):
        """
        Generate patch indices for spatial-reserved positional encoding.

        Returns:
        patch_indices (tensor): Tensor of patch indices in the format [patch_num, 3 (T, H, W)].
        """
        # Create an array for storing patch indices
        patch_indices = torch.tensor([[t * self.stride[0], h * self.stride[1], w * self.stride[2]]
                                      for t in range(self.t_patches)
                                      for h in range(self.h_patches)
                                      for w in range(self.w_patches)])

        return patch_indices


def rearrange_array(input_array, partition_lengths, index_tuples):
    # Validate the sum of partition lengths equals the number of features
    if sum(partition_lengths) != input_array.shape[1]:
        raise ValueError("Sum of partition lengths must equal the number of features in the input array.")

    # Validate the number of partitions matches the length of each tuple in index_tuples
    if any(len(t) != len(partition_lengths) for t in index_tuples):
        raise ValueError("Length of each index tuple must match the number of partitions.")

    # Initialize the output array
    output_array = np.zeros((len(index_tuples), input_array.shape[1]))

    # Define ending index for each partition
    partition_end_indices = np.cumsum(partition_lengths)

    # Extract and concatenate the specified rows and partitions
    for i, (start, end) in enumerate(zip([0] + list(partition_end_indices[:-1]), partition_end_indices)):
        indices = np.array([t[i] for t in index_tuples])
        output_array[:, start:end] = input_array[indices, start:end]

    return output_array

def calculate_correlation(list1, list2):
    """
    Calculate the Pearson correlation coefficient between two lists
    using PyTorch.

    Args:
    list1 (list): The first list of numerical values.
    list2 (list): The second list of numerical values.

    Returns:
    float: The Pearson correlation coefficient between the two lists.
    """
    # Convert lists to 1D PyTorch tensors
    tensor1 = torch.tensor(list1, dtype=torch.float32)
    tensor2 = torch.tensor(list2, dtype=torch.float32)

    # Ensure the tensors are 1D before stacking
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)

    # Stacking the tensors and calculating the correlation matrix
    stacked_tensors = torch.stack((tensor1_flat, tensor2_flat))
    correlation_matrix = torch.corrcoef(stacked_tensors)

    # The correlation coefficient is the element at (1, 0) of the matrix
    return correlation_matrix[1, 0].item()


def series_ids_permutation(Ds, length):
    """
    Shuffle the rows of the input array Ds independently for each column and adjust the shuffling
    based on the specified length. Populate Df with values drawn with or without replacement from Ds.
    Then, find the index of the first occurrence where values in Ds match those in Df for each column.

    Parameters:
    - Ds: A numpy array of shape (n, m) where n is the number of rows and m is the number of columns.
    - length: The number of rows to be generated in Df, which may be different from the number of rows in Ds.

    Returns:
    - Df: A numpy array of shape (length, m) with rows of Ds shuffled and adjusted based on the specified length.
    - syn_query: A numpy array of shape (length, m) containing the first occurrence index where Ds matches Df for each column.
    """
    nrows = Ds.shape[0]
    ncols = Ds.shape[1]
    np.random.seed()  # Ensures a different shuffle each time

    # Determine how to draw elements based on the specified length
    if length > nrows:
        # Draw with replacement
        Df = np.vstack([Ds[np.random.choice(nrows, length, replace=True), i] for i in range(ncols)]).T
    else:
        # Draw without replacement, similar to the original function but adjusted for the specified length
        Df = np.column_stack([Ds[np.random.permutation(nrows)[:length], i] for i in range(ncols)])

    # Initialize syn_query with NaNs
    syn_query = np.full((length, ncols), np.nan)

    # Populate syn_query
    for i in range(length):
        for j in range(ncols):
            cids = np.where(Ds[:, j] == Df[i, j])[0]
            if len(cids) > 0:
                syn_query[i, j] = cids[0]  # Python's 0-based indexing is maintained

    return Df, syn_query


def array_to_list_of_tuples(arr):
    """
    Convert a 2D numpy array into a list of tuples, where each tuple starts with the row index,
    followed by the values in that row, ensuring all elements are integers.

    Parameters:
    - arr: A 2D numpy array.

    Returns:
    - A list of tuples, where each tuple contains the row index followed by the values of each element in the row,
      with all elements converted to integers.
    """
    list_of_tuples = []
    for i in range(arr.shape[0]):  # Iterate over rows
        # Create a tuple that starts with the row index, then add all values from the row as integers
        row_tuple = tuple(int(value) for value in arr[i, :])
        list_of_tuples.append(row_tuple)

    return list_of_tuples




