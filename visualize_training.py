import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
from datetime import datetime

from datasets.simulated_target_rf import TargetMatrixGenerator
from datasets.simulated_dataset import MatrixDataset
from models.perceiver3d import Perceiver
from utils.training_procedure import load_checkpoint, forward_model
from utils.utils import DataVisualizer


def weightedsum_image_plot(output_image_np):
    plt.figure()
    plt.imshow(output_image_np, cmap='gray')  # Use cmap='gray' for grayscale images
    plt.title("Weighted Sum Image of RF")
    plt.xlabel("Width")
    plt.ylabel("Height")

def main():
    checkpoint_filename = 'Perceiver6timepoint_checkpoint_epoch_200'
    height = 20
    width = 24
    timepoint = 6

    checkpoint_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    # Construct the full path for the checkpoint file
    checkpoint_path = os.path.join(checkpoint_folder, f'{checkpoint_filename}.pth')

    # Generate a timestamp
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = 'visualize_model'
    # Construct the full path for the log file
    log_filename = os.path.join(saveprint_dir, f'{log_name}_training_log_{timestr}.txt')

    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
    device = torch.device("cuda")

    # Initialize the DataVisualizer
    visualizer_prog = DataVisualizer(savefig_dir, file_prefix='Plot_training_progress')
    visualizer_est_rf = DataVisualizer(savefig_dir, file_prefix='Estimate_RF')
    visualizer_est_rfstd = DataVisualizer(savefig_dir, file_prefix='Estimate_RF_std')
    visualizer_inout_corr = DataVisualizer(savefig_dir, file_prefix='Input_output_correlation')

    model = Perceiver(depth_dim=timepoint, height=height, width=width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch, model, optimizer, training_losses, validation_losses = load_checkpoint(checkpoint_path, model, optimizer, device)
    visualizer_prog.plot_and_save(None, plot_type='line', line1=training_losses, line2=validation_losses,
                                        xlabel='Epochs', ylabel='Loss')

    # Visual evaluation of results
    # Create the target matrix
    generator = TargetMatrixGenerator(mean=(0.1, -0.2), cov=np.array([[0.12, 0.05], [0.04, 0.03]]), device=device)

    # Generate the target matrix
    target_matrix = generator.create_3d_target_matrix(height, width, timepoint)
    logging.info(f"target_matrix size: {target_matrix.shape}")

    total_length = 10000  # Replace with your actual dataset length
    batch_size = 64  # Replace with your actual batch size

    dataset_test = MatrixDataset(target_matrix, total_length, device)
    sample_data, sample_label = dataset_test[0]
    logging.info(f"dataset size: {sample_data.shape}")
    output_image, weights, labels = forward_model(model, dataset_test, batch_size=batch_size)
    output_image_np = output_image.squeeze().cpu().numpy()
    visualizer_est_rf.plot_and_save(output_image_np, plot_type='3D_matrix', num_cols=5)
    visualizer_inout_corr.plot_and_save(None, plot_type='scatter', x_data=labels, y_data=weights,
                                                 xlabel='Labels', ylabel='Weights',
                                                 title='Relationship between Weights and Labels')
    output_image_np_std = np.std(output_image_np, axis=0)
    visualizer_est_rfstd.plot_and_save(output_image_np_std, plot_type='2D_matrix')

if __name__ == "__main__":
    main()