import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from datasets.simulated_target_rf import TargetMatrixGenerator
from datasets.simulated_dataset import MatrixDataset
from models.perceiver3d import Perceiver
from utils.training_procedure import load_checkpoint, forward_model
from utils.utils import DataVisualizer


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
    visualizer_prog = DataVisualizer(savefig_dir, file_prefix='Plot_training_progress')
    visualizer_est_rf = DataVisualizer(savefig_dir, file_prefix='Estimate_RF')
    visualizer_inout_corr = DataVisualizer(savefig_dir, file_prefix='Input_output_correlation')

    model = Perceiver().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch, model, optimizer, training_losses, validation_losses = load_checkpoint(checkpoint_path, model, optimizer, device)
    visualizer_prog.plot_and_save(None, plot_type='plot_line', x_data=training_losses, y_data=validation_losses,
                                        xlabel='Epochs', ylabel='Loss')

    # Visual evaluation of results
    # Create the target matrix
    generator = TargetMatrixGenerator(mean=(0.1, -0.2), cov=np.array([[0.12, 0.05], [0.04, 0.03]]), device=device)

    # Generate the target matrix
    target_matrix = generator.create_3d_target_matrix(30, 40, 20)

    total_length = 1000  # Replace with your actual dataset length
    batch_size = 64  # Replace with your actual batch size

    dataset_test = MatrixDataset(target_matrix, total_length)
    output_image, weights, labels = forward_model(model, dataset_test, batch_size=batch_size)
    output_image_np = output_image.squeeze().cpu().numpy()
    visualizer_est_rf.plot_and_save(output_image_np, plot_type='custom', custom_plot_func=weightedsum_image_plot)
    visualizer_inout_corr.plot_and_save(None, plot_type='scatter', x_data=labels, y_data=weights,
                                                 xlabel='Labels', ylabel='Weights',
                                                 title='Relationship between Weights and Labels')

if __name__ == "__main__":
    main()