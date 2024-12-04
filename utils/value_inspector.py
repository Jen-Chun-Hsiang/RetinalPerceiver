import os
import matplotlib.pyplot as plt
import torch
import numpy as np


def save_distributions(train_loader, n, folder_name, file_name, logging=None):
    """
    Accumulates values from batches, plots 1D distributions, and saves the plots.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        n (int): Number of batches to process.
        folder_name (str): Directory where the plots will be saved.
        file_name (str): Base name for the saved plot file.
    """
    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Accumulators for each variable
    all_random_matrix = []
    all_output_value = []

    # Collect data over n batches
    with torch.no_grad():
        for batch_idx, (random_matrix, output_value, _) in enumerate(train_loader):
            if batch_idx >= n:
                break  # Stop after processing n batches

            # Flatten tensors and accumulate
            all_random_matrix.append(random_matrix.view(-1).cpu().numpy())
            all_output_value.append(output_value.view(-1).cpu().numpy())
            if logging is not None:
                logging.info(f'batch_idx: {batch_idx} \n')

    # Concatenate accumulated values
    all_random_matrix = np.concatenate(all_random_matrix)
    all_output_value = np.concatenate(all_output_value)

    # Plot separate distributions in subplots
    plt.figure(figsize=(12, 6))

    # Subplot 1: Random Matrix
    plt.subplot(1, 2, 1)
    plt.hist(all_random_matrix, bins=50, alpha=0.7, color='blue')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Random Matrix")

    # Subplot 2: Output Value
    plt.subplot(1, 2, 2)
    plt.hist(all_output_value, bins=50, alpha=0.7, color='green')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Output Value")

    # Adjust layout and save the plot
    plt.tight_layout()
    save_path = os.path.join(folder_name, file_name)
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to {save_path}")
