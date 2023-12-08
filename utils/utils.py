import torch
import os
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

    def _plot_3d_matrix(self, data, num_cols):
        num_rows = data.shape[0] // num_cols + int(data.shape[0] % num_cols > 0)
        plt.figure(figsize=(15, num_rows * 3))
        for i in range(data.shape[0]):
            ax = plt.subplot(num_rows, num_cols, i + 1)
            image = ax.imshow(data[i], cmap='viridis', vmin=-0.4, vmax=0.5)
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
