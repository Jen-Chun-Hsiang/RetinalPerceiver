import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
from datetime import datetime
from scipy.io import savemat

from datasets.simulated_target_rf import MultiTargetMatrixGenerator, ParameterGenerator
from datasets.simulated_dataset import MultiMatrixDataset
from models.perceiver3d import RetinalPerceiverIO
from models.cnn3d import RetinalPerceiverIOWithCNN
from utils.training_procedure import CheckpointLoader, forward_model
from utils.utils import DataVisualizer, SeriesEncoder, rearrange_array, calculate_correlation, series_ids_permutation
from utils.utils import series_ids_permutation_uni
from utils.utils import array_to_list_of_tuples
from utils.utils import plot_and_save_3d_matrix_with_timestamp as plot3dmat
from utils.result_analysis import find_connected_center, pairwise_mult_sum

def weightedsum_image_plot(output_image_np):
    plt.figure()
    plt.imshow(output_image_np, cmap='gray')  # Use cmap='gray' for grayscale images
    plt.title("Weighted Sum Image of RF")
    plt.xlabel("Width")
    plt.ylabel("Height")


def main():
    # Common variables for all configurations
    stimulus_type = 'SIMPlugIn_09232403'
    epoch_end = 150
    perm_cols = [0, 1]  # (0, 1) for linear (0, 1, 2) for masking (0, 1, 2, 3) for num_cell
    rand_sample_cols = [2, 3]
    num_rand_sample = 12
    is_full_figure_draw = False

    # Define all possible configurations
    configurations = {
        '1': {'is_cross_level': False, 'is_weight_in_label': False},
        '2': {'is_cross_level': True, 'is_weight_in_label': False},
        '3': {'is_cross_level': True, 'is_weight_in_label': True},
    }

    # Specify which configurations to run
    config_ids = ['1', '2', '3']  # Adjust this list to include the config IDs you want to run
    for key in config_ids:
        if key in configurations:
            config = configurations[key]
            is_continue = run_configuration(
                stimulus_type=stimulus_type,
                epoch_end=epoch_end,
                perm_cols=perm_cols,  # Use specific perm_cols from config
                is_full_figure_draw=is_full_figure_draw,
                is_cross_level=config['is_cross_level'],
                is_weight_in_label=config['is_weight_in_label'],
                rand_sample_cols=rand_sample_cols,
                num_rand_sample=num_rand_sample,

            )
        else:
            print(f"No configuration found for key: {key}")


def run_configuration(stimulus_type, epoch_end, perm_cols, is_full_figure_draw, is_cross_level, is_weight_in_label,
                      rand_sample_cols, num_rand_sample):
    checkpoint_filename = f'PerceiverIO_{stimulus_type}_checkpoint_epoch_{epoch_end}'

    # default parameters
    total_length = 10000  # Replace with your actual dataset length
    permute_series_length = 240
    batch_size = 32  # Replace with your actual batch size
    checkpoint_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    savedata_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Data/'
    savemat_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Matfiles/'

    # Construct the full path for the checkpoint file
    checkpoint_path = os.path.join(checkpoint_folder, f'{checkpoint_filename}.pth')

    # Generate a timestamp
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = 'visualize_model'
    # Construct the full path for the log file
    log_filename = os.path.join(saveprint_dir, f'{checkpoint_filename}_training_log_{timestr}.txt')

    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    #
    visualizer_prog = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Training_progress')

    # Load the training and model parameters
    checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path, device=device)
    training_losses = checkpoint_loader.load_training_losses()
    validation_losses = checkpoint_loader.load_validation_losses()
    validation_contra_losses = checkpoint_loader.load_validation_contra_losses()
    logging.info(f'training_losses:{training_losses} \n')
    logging.info(f'validation_losses:{validation_losses} \n')
    logging.info(f'validation_contra_losses:{validation_contra_losses} \n')
    ValueError(f"Temporal stop {timestr}! (remove after use)")
    visualizer_prog.plot_and_save(None, plot_type='line', line1=training_losses, line2=validation_losses,
                                  xlabel='Epochs', ylabel='Loss')
    args = checkpoint_loader.load_args()
    config_module = f"configs.sims.{args.config_name}"
    config = __import__(config_module, fromlist=[''])

    if not hasattr(args, 'rng_seed') or args.rng_seed is None:
        args.rng_seed = 42

    query_table = getattr(config, 'query_table', None)
    sf_param_table = getattr(config, 'sf_param_table', None)
    tf_param_table = getattr(config, 'tf_param_table', None)
    # Generate param_list
    if args.rng_seed is None:
        raise RuntimeError("Random seed it not assigned at the training.")
    parameter_generator = ParameterGenerator(sf_param_table, tf_param_table, seed=args.rng_seed)
    param_lists, series_ids = parameter_generator.generate_parameters(query_table)
    # syn_param_lists = parameter_generator.generate_parameters_from_query_list(series_ids)
    # syn_param_list = np.array(syn_param_list, dtype=float)
    # syn_param_lists = [x if x is not None else np.nan for x in syn_param_lists]
    # print(f'syn_param_list type: {type(syn_param_lists)}')
    # print(f'syn_param_lists: {syn_param_lists[0]}')
    # print(f'param_lists : {param_lists[0]}')
    # print(f'syn_param_lists: {syn_param_lists[100]}')
    # print(f'param_lists : {param_lists[100]}')
    # print(f'syn_param_lists: {syn_param_lists[-1]}')
    # print(f'param_lists : {param_lists[-1]}')
    # raise RuntimeError("Script stopped after saving outputs.")

    # Save to .mat file
    # savemat(os.path.join(savemat_dir, 'sim_multi_list_10022401.mat'),
    #        {"param_list": param_lists, "series_ids": series_ids, "syn_param_lists": syn_param_lists})
    # raise RuntimeError("Script stopped after saving outputs.")


    # Encode series_ids into query arrays
    max_values = getattr(config, 'max_values', None)
    # skip_encoding = getattr(config, 'skip_encoding', None)
    encoding_type = getattr(config, 'encoding_type', None)
    lengths = getattr(config, 'lengths', None)
    shuffle_components = getattr(config, 'shuffle_components', None)
    query_encoder = SeriesEncoder(max_values, lengths, encoding_method=args.encoding_method, encoding_type=encoding_type,
                                  shuffle_components=shuffle_components)
    logging.info(f'series_ids:{series_ids} \n')
    query_arrays = query_encoder.encode(series_ids)
    logging.info(f'query_arrays shape:{query_arrays.shape} \n')

    # query_permutator = None

    # Initialize the DataVisualizer
    visualizer_est_rf = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Estimate_RF')
    visualizer_est_rfstd = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Estimate_RF_std')
    visualizer_inout_corr = DataVisualizer(savefig_dir, file_prefix=f'{stimulus_type}_Input_output_correlation')


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
                                           use_layer_norm=args.use_layer_norm, num_bands=args.num_band,
                                           conv3d_out_channels=args.conv3d_out_channels,
                                           conv2_out_channels=args.conv2_out_channels,
                                           conv2_1st_layer_kernel=args.conv2_1st_layer_kernel,
                                           conv2_2nd_layer_kernel=args.conv2_2nd_layer_kernel,
                                           device=device,).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model, optimizer = checkpoint_loader.load_checkpoint(model, optimizer)

    if is_cross_level:
        logging.info(f'series_ids example 1:{series_ids[0]} \n')
        syn_series_ids = series_ids_permutation_uni(np.array(series_ids), perm_cols, rand_sample_cols=rand_sample_cols,
                                                    num_rand_sample=num_rand_sample)
        logging.info(f'syn_series_ids example 1:{syn_series_ids[0]} \n')
        logging.info(f'syn_series_ids example -1:{syn_series_ids[-1]} \n')

        param_lists = parameter_generator.generate_parameters_from_query_list(syn_series_ids)
        # print(f'syn_param_lists 1: {param_lists[0]}')
        # print(f'syn_param_lists 2: {param_lists[201]}')
        # print(f'syn_param_lists 3: {param_lists[401]}')
        # print(f'syn_param_lists 4: {param_lists[1201]}')
        # print(f'syn_param_lists 5: {param_lists[-1]}')
        # raise RuntimeError("Script stopped after saving outputs.")
        syn_query_index = query_encoder.encode(syn_series_ids)

        logging.info(f'syn_query_index example 1:{syn_query_index[0, :]} \n')
        query_arrays = syn_query_index
        cross_level_flag = 'Interpolation'
    else:
        syn_series_ids = np.array([])
        syn_query_index = np.array([])
        cross_level_flag = 'Data'

    if is_weight_in_label:
        label_flag = 'Label'
    else:
        label_flag = 'Model'

    savedata_filename_npz = os.path.join(savedata_dir, f'{checkpoint_filename}_data_{cross_level_flag}_{label_flag}.npz')
    savedata_filename_mat = os.path.join(savedata_dir, f'{checkpoint_filename}_data_{cross_level_flag}_{label_flag}.mat')

    presented_cell_ids = list(range(query_arrays.shape[0]))
    logging.info(f"presented_cell_ids: {presented_cell_ids}")
    num_cols = 5
    corrcoef_vals = np.zeros((query_arrays.shape[0], 1))
    rf_spatial_array_list = []
    rf_spatial_peak_array_list = []
    rf_spatial_trough_array_list = []
    ii = 0
    for presented_cell_id in presented_cell_ids:
        query_array = query_arrays[presented_cell_id:presented_cell_id+1, :]
        logging.info(f'query_encoder {presented_cell_id}:{query_array.shape} \n')
        # Use param_list in MultiTargetMatrixGenerator
        param_list = param_lists[presented_cell_id]
        multi_target_gen = MultiTargetMatrixGenerator(param_list)
        target_matrix = multi_target_gen.create_3d_target_matrices(
            input_height=args.input_height, input_width=args.input_width, input_depth=args.input_depth)

        # Initialize the dataset with the device
        # plot3dmat(target_matrix[0, :, :, :], num_cols, savefig_dir, file_prefix=f'plot_3D_matrix_{cross_level_flag}_{presented_cell_id}')
        dataset_test = MultiMatrixDataset(target_matrix, length=total_length, device=device, combination_set=[1],
                                     add_noise=args.add_noise, noise_level=args.noise_level, use_relu=args.use_relu,
                                     output_offset=args.output_offset)

        output_image, weights, labels = forward_model(model, dataset_test, query_array=query_array, batch_size=batch_size,
                                                      use_matrix_index=False, is_weight_in_label=is_weight_in_label)
        # raise RuntimeError("Script stopped after saving outputs.")
        output_image_np = output_image.squeeze().cpu().numpy()
        output_image_np_std = np.std(output_image_np, axis=0)
        output_image_np_std = output_image_np_std / output_image_np_std.sum()
        rf_center = find_connected_center(output_image_np_std)
        rf_temporal = pairwise_mult_sum(output_image_np_std, output_image_np)
        max_indices = np.where(rf_temporal == max(rf_temporal))[0]
        # Check if max_indices is empty, if so, create a zero array of the same shape as a slice of output_image_np
        if max_indices.size == 0:
            rf_spatial_peak = np.zeros_like(output_image_np[0, :, :])
        else:
            rf_spatial_peak = np.squeeze(output_image_np[max_indices, :, :])

        min_indices = np.where(rf_temporal == min(rf_temporal))[0]
        if min_indices.size == 0:
            rf_spatial_trough = np.zeros_like(output_image_np[0, :, :])
        else:
            rf_spatial_trough = np.squeeze(output_image_np[min_indices, :, :])

        if ii == 0:
            rf_center_array = rf_center.reshape(1, -1)
            rf_temporal_array = rf_temporal.reshape(1, -1)
        else:
            rf_center_array = np.concatenate((rf_center_array, rf_center.reshape(1, -1)), axis=0)
            rf_temporal_array = np.concatenate((rf_temporal_array, rf_temporal.reshape(1, -1)), axis=0)
        rf_spatial_array_list.append(output_image_np_std)
        rf_spatial_peak_array_list.append(rf_spatial_peak)
        rf_spatial_trough_array_list.append(rf_spatial_trough)

        if is_full_figure_draw:
            visualizer_est_rf.plot_and_save(output_image_np, plot_type='3D_matrix', num_cols=5)
            if is_weight_in_label is False:
                visualizer_inout_corr.plot_and_save(None, plot_type='scatter', x_data=labels, y_data=weights,
                                                    xlabel='Labels', ylabel='Weights',
                                                    title='Relationship between Weights and Labels')
            visualizer_est_rfstd.plot_and_save(output_image_np_std, plot_type='2D_matrix')

        if is_weight_in_label is False:
            corrcoef_vals[ii, :] = calculate_correlation(labels, weights)
        ii += 1

    rf_spatial_array = np.stack(rf_spatial_array_list, axis=2)
    rf_spatial_peak_array = np.stack(rf_spatial_peak_array_list, axis=2)
    rf_spatial_trough_array = np.stack(rf_spatial_trough_array_list, axis=2)
    logging.info(f'correlation coefficient: {corrcoef_vals} \n')

    np.savez(savedata_filename_npz,
             rf_center_array=rf_center_array, rf_temporal_array=rf_temporal_array,
             rf_spatial_array=rf_spatial_array, corrcoef_vals=corrcoef_vals,
             query_arrays=query_arrays, series_ids=series_ids,
             syn_series_ids=syn_series_ids, syn_query_index=syn_query_index,
             rf_spatial_peak_array=rf_spatial_peak_array, rf_spatial_trough_array=rf_spatial_trough_array)

    # Save the dictionary as a .mat file
    savemat(savedata_filename_mat, {
        'rf_center_array': rf_center_array, 'rf_temporal_array': rf_temporal_array,
        'corrcoef_vals': corrcoef_vals, 'rf_spatial_array': rf_spatial_array,
        'query_arrays': query_arrays, 'series_ids': series_ids,
        'syn_series_ids': syn_series_ids, 'syn_query_index': syn_query_index,
        'rf_spatial_peak_array': rf_spatial_peak_array, 'rf_spatial_trough_array': rf_spatial_trough_array
    })


if __name__ == "__main__":
    main()