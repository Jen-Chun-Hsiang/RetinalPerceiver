import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
from datetime import datetime
from scipy.io import savemat

from datasets.simulated_target_rf import MultiTargetMatrixGenerator, CellClassLevel, ExperimentalLevel, IntegratedLevel
from datasets.simulated_dataset import MultiMatrixDataset
from models.perceiver3d import RetinalPerceiverIO
from models.cnn3d import RetinalPerceiverIOWithCNN
from utils.training_procedure import CheckpointLoader, forward_model
from utils.utils import DataVisualizer, SeriesEncoder, rearrange_array, calculate_correlation
from utils.utils import plot_and_save_3d_matrix_with_timestamp as plot3dmat


def weightedsum_image_plot(output_image_np):
    plt.figure()
    plt.imshow(output_image_np, cmap='gray')  # Use cmap='gray' for grayscale images
    plt.title("Weighted Sum Image of RF")
    plt.xlabel("Width")
    plt.ylabel("Height")

def main():
    stimulus_type = '200ktl2011ks111sd'

    epoch_end = 200
    is_cross_level = False
    is_full_figure_draw = True
    checkpoint_filename = f'PerceiverIO_{stimulus_type}_checkpoint_epoch_{epoch_end}'
    examine_list = [(0, 0, 15), (15, 15, 0), (34, 34, 3), (3, 3, 34)]  # List of tuples for row selection

    # default parameters
    total_length = 10000  # Replace with your actual dataset length
    batch_size = 32  # Replace with your actual batch size
    checkpoint_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    savedata_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Data/'
    # Construct the full path for the checkpoint file
    checkpoint_path = os.path.join(checkpoint_folder, f'{checkpoint_filename}.pth')

    # Generate a timestamp
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = 'visualize_model'
    # Construct the full path for the log file
    log_filename = os.path.join(saveprint_dir, f'{checkpoint_filename}_training_log_{timestr}.txt')
    savedata_filename_npy = os.path.join(savedata_dir, f'{checkpoint_filename}_data.npy')
    savedata_filename_mat = os.path.join(savedata_dir, f'{checkpoint_filename}_data.mat')

    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    # Create cells and cell classes
    cell_class1_layout1 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                         sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class1_layout2 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                         sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=8, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class1_layout3 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                         sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=7, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class1_layout4 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                         sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=9, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class1_layout5 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                         sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=5, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

    cell_class2_layout1 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                         sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                         sf_weight_surround=0.3, num_cells=8, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class2_layout2 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                         sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                         sf_weight_surround=0.3, num_cells=11, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class2_layout3 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                         sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                         sf_weight_surround=0.3, num_cells=13, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class2_layout4 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                         sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                         sf_weight_surround=0.3, num_cells=12, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class2_layout5 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                         sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                         sf_weight_surround=0.3, num_cells=10, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

    cell_class3_layout1 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                         sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                         sf_weight_surround=0.5, num_cells=16, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class3_layout2 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                         sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                         sf_weight_surround=0.5, num_cells=14, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class3_layout3 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                         sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                         sf_weight_surround=0.5, num_cells=14, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class3_layout4 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                         sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                         sf_weight_surround=0.5, num_cells=15, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class3_layout5 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                         sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                         sf_weight_surround=0.5, num_cells=13, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

    cell_class4_layout1 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                         sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class4_layout2 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                         sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class4_layout3 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                         sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=7, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class4_layout4 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                         sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=5, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class4_layout5 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                         sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                         sf_weight_surround=0.5, num_cells=7, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

    cell_class5_layout1 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                         sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                         sf_weight_surround=0.4, num_cells=9, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class5_layout2 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                         sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                         sf_weight_surround=0.4, num_cells=10, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class5_layout3 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                         sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                         sf_weight_surround=0.4, num_cells=9, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class5_layout4 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                         sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                         sf_weight_surround=0.4, num_cells=10, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class5_layout5 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                         sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                         sf_weight_surround=0.4, num_cells=11, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

    cell_class6_layout1 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                         sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                         sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class6_layout2 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                         sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                         sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class6_layout3 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                         sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                         sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class6_layout4 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                         sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                         sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
    cell_class6_layout5 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                         sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                         sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

    # Create experimental level with cell classes
    experimental = ExperimentalLevel(tf_weight_surround=0.2, tf_sigma_center=0.05,
                                     tf_sigma_surround=0.12, tf_mean_center=0.08,
                                     tf_mean_surround=0.12, tf_weight_center=1,
                                     tf_offset=0, cell_classes=[cell_class1_layout1, cell_class2_layout1,
                                                                cell_class3_layout1, cell_class4_layout1,
                                                                cell_class5_layout1, cell_class6_layout1])

    experimental2 = ExperimentalLevel(tf_weight_surround=0.3, tf_sigma_center=0.04,
                                      tf_sigma_surround=0.10, tf_mean_center=0.07,
                                      tf_mean_surround=0.10, tf_weight_center=1,
                                      tf_offset=0, cell_classes=[cell_class1_layout2, cell_class2_layout2,
                                                                 cell_class3_layout2, cell_class4_layout2,
                                                                 cell_class5_layout2, cell_class6_layout2])

    experimental3 = ExperimentalLevel(tf_weight_surround=0.4, tf_sigma_center=0.03,
                                      tf_sigma_surround=0.09, tf_mean_center=0.06,
                                      tf_mean_surround=0.11, tf_weight_center=1,
                                      tf_offset=0, cell_classes=[cell_class1_layout3, cell_class2_layout3,
                                                                 cell_class3_layout3, cell_class4_layout3,
                                                                 cell_class5_layout3, cell_class6_layout3])

    experimental4 = ExperimentalLevel(tf_weight_surround=0.1, tf_sigma_center=0.08,
                                      tf_sigma_surround=0.2, tf_mean_center=0.15,
                                      tf_mean_surround=0.2, tf_weight_center=1,
                                      tf_offset=0, cell_classes=[cell_class1_layout4, cell_class2_layout4,
                                                                 cell_class3_layout4, cell_class4_layout4,
                                                                 cell_class5_layout4, cell_class6_layout4])

    experimental5 = ExperimentalLevel(tf_weight_surround=-0.1, tf_sigma_center=0.07,
                                      tf_sigma_surround=0.18, tf_mean_center=0.13,
                                      tf_mean_surround=0.18, tf_weight_center=-1,
                                      tf_offset=0, cell_classes=[cell_class1_layout5, cell_class2_layout5,
                                                                 cell_class3_layout5, cell_class4_layout5,
                                                                 cell_class5_layout5, cell_class6_layout5])

    # Create integrated level with experimental levels
    integrated_list = IntegratedLevel([experimental, experimental2, experimental3])

    # Generate param_list
    param_lists, series_ids = integrated_list.generate_combined_param_list()

    # Encode series_ids into query arrays
    max_values = {'Experiment': 100, 'Type': 100, 'Cell': 10000}
    lengths = {'Experiment': 6, 'Type': 6, 'Cell': 24}
    shuffle_components = ['Cell']
    query_encoder = SeriesEncoder(max_values, lengths, shuffle_components=shuffle_components)
    query_arrays = query_encoder.encode(series_ids)
    logging.info(f'query_arrays example 1:{query_arrays.shape} \n')

    # Initialize the DataVisualizer
    visualizer_prog = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Training_progress')
    visualizer_est_rf = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Estimate_RF')
    visualizer_est_rfstd = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Estimate_RF_std')
    visualizer_inout_corr = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Input_output_correlation')

    checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path, device=device)
    args = checkpoint_loader.load_args()

    if args.model == 'RetinalPerceiver':
        model = RetinalPerceiverIO(input_dim=args.input_channels, latent_dim=args.hidden_size,
                                   output_dim=args.output_size, num_latents=args.num_latent, heads=args.num_head,
                                   depth=args.num_iter, query_dim=query_arrays.shape[1], depth_dim=args.input_depth,
                                   height=args.input_height, width=args.input_width, num_bands=args.num_band,
                                   device=device, use_layer_norm=args.use_layer_norm, kernel_size=args.kernel_size,
                                   stride=args.stride, concatenate_positional_encoding=args.concatenate_positional_encoding,
                                   use_phase_shift=args.use_phase_shift, use_dense_frequency=args.use_dense_frequency)

    elif args.model == 'RetinalCNN':
         model = RetinalPerceiverIOWithCNN(input_depth=args.input_depth, input_height=args.input_height,
                                           input_width=args.input_width, output_dim=args.output_size,
                                           latent_dim=args.hidden_size, query_dim=query_arrays.shape[1],
                                           num_latents=args.num_latent, heads=args.num_head,
                                           use_layer_norm=args.use_layer_norm, device=device,
                                           num_bands=args.num_band, conv3d_out_channels=args.conv3d_out_channels,
                                           conv2_out_channels=args.conv2_out_channels).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model, optimizer = checkpoint_loader.load_checkpoint(model, optimizer)
    training_losses, validation_losses = checkpoint_loader.get_training_losses(), checkpoint_loader.get_validation_losses()
    visualizer_prog.plot_and_save(None, plot_type='line', line1=training_losses, line2=validation_losses,
                                  xlabel='Epochs', ylabel='Loss')

    if is_cross_level:
        query_partition_lengths = tuple(lengths.values())
        query_arrays = rearrange_array(query_arrays, query_partition_lengths, examine_list)
        cross_level_flag = 'Interpolation'
    else:
        cross_level_flag = 'Data'

    presented_cell_ids = list(range(query_arrays.shape[0]))

    num_cols = 5
    corrcoef_vals = np.zeros((query_arrays.shape[0], 1))
    ii = 0
    for presented_cell_id in presented_cell_ids:
        query_array = query_arrays[presented_cell_id:presented_cell_id+1, :]
        logging.info(f'query_encoder example 1:{query_array.shape} \n')
        # Use param_list in MultiTargetMatrixGenerator
        param_list = param_lists[presented_cell_id]
        multi_target_gen = MultiTargetMatrixGenerator(param_list)
        target_matrix = multi_target_gen.create_3d_target_matrices(
            input_height=args.input_height, input_width=args.input_width, input_depth=args.input_depth)

        logging.info(f'target matrix: {target_matrix.shape}  \n')

        # Initialize the dataset with the device

        plot3dmat(target_matrix[0, :, :, :], num_cols, savefig_dir, file_prefix=f'plot_3D_matrix_{cross_level_flag}_{presented_cell_id}')
        #dataset_test = MultiMatrixDataset(target_matrix, length=total_length, device=device, combination_set=[1])
        dataset_test = MultiMatrixDataset(target_matrix, length=args.total_length, device=device, combination_set=[1],
                                     add_noise=args.add_noise, noise_level=args.noise_level, use_relu=args.use_relu,
                                     output_offset=args.output_offset)

        sample_data, sample_label, sample_index = dataset_test[0]
        logging.info(f"dataset size: {sample_data.shape}")
        output_image, weights, labels = forward_model(model, dataset_test, query_array=query_array, batch_size=batch_size)

        if is_full_figure_draw:
            output_image_np = output_image.squeeze().cpu().numpy()
            visualizer_est_rf.plot_and_save(output_image_np, plot_type='3D_matrix', num_cols=5)
            visualizer_inout_corr.plot_and_save(None, plot_type='scatter', x_data=labels, y_data=weights,
                                                xlabel='Labels', ylabel='Weights',
                                                title='Relationship between Weights and Labels')
            output_image_np_std = np.std(output_image_np, axis=0)
            visualizer_est_rfstd.plot_and_save(output_image_np_std, plot_type='2D_matrix')

        corrcoef_vals[ii, :] = calculate_correlation(labels, weights)
        ii += 1

    logging.info(f'correlation coefficient: {corrcoef_vals} \n')
    np.save(savedata_filename_npy, corrcoef_vals)

    # Save the dictionary as a .mat file
    savemat(savedata_filename_mat, {'corrcoef_vals': corrcoef_vals})

if __name__ == "__main__":
    main()