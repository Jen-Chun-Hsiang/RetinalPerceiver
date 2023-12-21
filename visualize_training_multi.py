import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
from datetime import datetime

from datasets.simulated_target_rf import MultiTargetMatrixGenerator, CellClassLevel, ExperimentalLevel, IntegratedLevel
from datasets.simulated_dataset import MultiMatrixDataset
from models.perceiver3d import RetinalPerceiverIO
from models.cnn3d import RetinalPerceiverIOWithCNN
from utils.training_procedure import load_checkpoint, forward_model
from utils.utils import DataVisualizer, SeriesEncoder


def weightedsum_image_plot(output_image_np):
    plt.figure()
    plt.imshow(output_image_np, cmap='gray')  # Use cmap='gray' for grayscale images
    plt.title("Weighted Sum Image of RF")
    plt.xlabel("Width")
    plt.ylabel("Height")

def main():
    # experiment specific parameters
    presented_cell_id = 0
    height = 20
    width = 24
    time_point = 20
    query_dim = 6
    hidden_size = 128
    num_latents = 16
    conv3d_out_channels = 10  # default 1
    use_layer_norm = True
    stimulus_type = 'combo50000tfsfstim123LYnorm1cGelu'
    model_type = 'RetinalPerceiver'
    checkpoint_filename = f'PerceiverIO_{time_point}tp{stimulus_type}_checkpoint_epoch_200'

    # default parameters
    total_length = 10000  # Replace with your actual dataset length
    batch_size = 64  # Replace with your actual batch size
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
    visualizer_prog = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Training_progress')
    visualizer_est_rf = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Estimate_RF')
    visualizer_est_rfstd = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Estimate_RF_std')
    visualizer_inout_corr = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Input_output_correlation')

    if model_type == 'RetinalPerceiver':
        model = RetinalPerceiverIO(query_dim=query_dim, depth_dim=time_point, height=height, width=width,
                                   device=device, use_layer_norm=use_layer_norm, latent_dim=hidden_size,
                                   num_latents=num_latents)
    elif model_type == 'RetinalCNN':
        model = RetinalPerceiverIOWithCNN(input_depth=time_point, input_height=height,
                                          input_width=width, latent_dim=hidden_size,
                                          query_dim=query_dim, num_latents=num_latents,
                                          use_layer_norm=use_layer_norm, device=device,
                                          conv3d_out_channels=conv3d_out_channels)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch, model, optimizer, training_losses, validation_losses = load_checkpoint(checkpoint_path, model,
                                                                                        optimizer, device)
    visualizer_prog.plot_and_save(None, plot_type='line', line1=training_losses, line2=validation_losses,
                                        xlabel='Epochs', ylabel='Loss')

    # Create cells and cell classes
    cell_class1 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]),
                                 sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                 sf_weight_surround=0.5, num_cells=1, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

    # Create experimental level with cell classes
    experimental = ExperimentalLevel(tf_weight_surround=0.2, tf_sigma_center=0.05,
                                     tf_sigma_surround=0.12, tf_mean_center=0.08,
                                     tf_mean_surround=0.12, tf_weight_center=1,
                                     tf_offset=0, cell_classes=[cell_class1])

    # Create integrated level with experimental levels
    integrated_list = IntegratedLevel([experimental])

    # Generate param_list
    param_list, series_ids = integrated_list.generate_combined_param_list()
    logging.info(f'parameter list:{param_list} \n')
    param_list = param_list[presented_cell_id]
    logging.info(f'parameter list (selected):{param_list} \n')
    # Encode series_ids into query arrays
    max_values = {'Experiment': 10, 'Type': 10, 'Cell': 10}
    lengths = {'Experiment': 2, 'Type': 2, 'Cell': 2}
    shuffle_components = ['Cell']
    query_encoder = SeriesEncoder(max_values, lengths, shuffle_components=shuffle_components)
    query_array = query_encoder.encode(series_ids)
    query_array = query_array[presented_cell_id:presented_cell_id+1, :]
    logging.info(f'query_array size:{query_array.shape} \n')
    # Use param_list in MultiTargetMatrixGenerator
    multi_target_gen = MultiTargetMatrixGenerator(param_list)
    target_matrix = multi_target_gen.create_3d_target_matrices(
        input_height=height, input_width=width, input_depth=time_point)

    logging.info(f'target matrix: {target_matrix.shape}  \n')

    # Initialize the dataset with the device
    dataset_test = MultiMatrixDataset(target_matrix, length=total_length, device=device,
                                 combination_set=[1])

    sample_data, sample_label, sample_index = dataset_test[0]
    logging.info(f"dataset size: {sample_data.shape}")
    output_image, weights, labels = forward_model(model, dataset_test, query_array=query_array, batch_size=batch_size)
    output_image_np = output_image.squeeze().cpu().numpy()
    visualizer_est_rf.plot_and_save(output_image_np, plot_type='3D_matrix', num_cols=5)
    visualizer_inout_corr.plot_and_save(None, plot_type='scatter', x_data=labels, y_data=weights,
                                                 xlabel='Labels', ylabel='Weights',
                                                 title='Relationship between Weights and Labels')
    output_image_np_std = np.std(output_image_np, axis=0)
    visualizer_est_rfstd.plot_and_save(output_image_np_std, plot_type='2D_matrix')

if __name__ == "__main__":
    main()