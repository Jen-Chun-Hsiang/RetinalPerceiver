import torch
import torch.optim as optim
from models.perceiver3d import Perceiver
from utils.training_procedure import load_checkpoint
from utils.utils import DataVisualizer
import matplotlib.pyplot as plt

def weightedsum_image_plot(output_image_np):
    plt.figure()
    plt.imshow(output_image_np, cmap='gray')  # Use cmap='gray' for grayscale images
    plt.title("Weigthed sum image of RF")
    plt.xlabel("Width")
    plt.ylabel("Height")

def main():
    # Existing main function setup

    checkpoint_filename = ''
    checkpoint_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")

    # If CUDA is available, continue with the rest of the script
    device = torch.device("cuda")

    # Initialize the DataVisualizer
    result_dir = 'path/to/save/directory'  # Replace with your actual directory
    visualizer_prog = DataVisualizer(savefig_dir, file_prefix='Plot_training_progress')
    visualizer_est_rf = DataVisualizer(savefig_dir, file_prefix='Estimate_RF')
    visualizer_inout_corr = DataVisualizer(savefig_dir, file_prefix='Input_output_correlation')

    model = Perceiver().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch, model, optimizer, training_losses, validation_losses = load_checkpoint(checkpoint_path, model, optimizer, device)

    plot_losses(training_losses, validation_losses, start_epoch)

    # Visual evaluation of results
    target_matrix = None  # Replace with your actual target matrix
    total_length = None  # Replace with your actual dataset length
    batch_size = None  # Replace with your actual batch size

    dataset_test = MatrixDataset(target_matrix, total_length)
    output_image, weights, labels = forward_model(model, dataset_test, batch_size=batch_size)
    output_image_np = output_image.squeeze().cpu().numpy()
    visualizer_est_rf.plot_and_save(output_image_np, plot_type='custom', custom_plot_func=weightedsum_image_plot)
    visualizer_inout_corr.plot_and_save(output_image_np, plot_type='custom', custom_plot_func=weightedsum_image_plot)

if __name__ == "__main__":
    main()