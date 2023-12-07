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


