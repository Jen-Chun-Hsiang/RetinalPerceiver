from datasets.preprocessing import PNGToTensorConverter
from datasets.preprocessing import process_experiment_folders


def convert_png_to_ph_overwrite(experiment_id, scale_factor, padding_size, set_name, save_type):
    # (1) Convert png to ph
    convert_root = f"/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/{set_name}/Stimulus" \
                   f"/experiment_{experiment_id}/ "
    converter = PNGToTensorConverter(convert_root, overwrite=True, scale_factor=scale_factor, padding_size=padding_size,
                                     save_type=save_type)
    converter.start_conversion()


def convert_png_to_ph_addnew(experiment_id, scale_factor, padding_size, set_name, save_type):
    # (2) Skip conversion if .pt file exists
    convert_root = f"/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/{set_name}/Stimulus" \
                   f"/experiment_{experiment_id}/ "
    converter = PNGToTensorConverter(convert_root, overwrite=False, scale_factor=scale_factor, padding_size=padding_size, save_type=save_type)
    converter.start_conversion()


def generate_session_in_hdf5(experiment_id, scale_factor, padding_size, set_name):
    # (3) Gather image into hdf5 file
    root_folder = f"/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/{set_name}/Stimulus" \
                  f"/experiment_{experiment_id}/ "
    process_experiment_folders(root_folder, scale_factor=scale_factor, padding_size=padding_size)


def execute_task(task_id, experiment_ids, scale_factor, padding_size, set_name, save_type):
    task_functions = {
        1: convert_png_to_ph_overwrite,
        2: convert_png_to_ph_addnew,
        3: generate_session_in_hdf5,
    }
    func = task_functions.get(task_id)
    if func is None:
        raise ValueError("No task found for argument: {}".format(task_id))
    for experiment_id in experiment_ids:
        if task_id in [1, 2]:  # Tasks that require save_type
            func(experiment_id, scale_factor, padding_size, set_name, save_type)
        else:  # Task 3 does not require save_type
            func(experiment_id, scale_factor, padding_size, set_name)


if __name__ == "__main__":
    run_task_id = 1
    scale_factor = 1 / 2.5
    padding_size = 5
    experiment_ids = [4, 5, 6, 8, 9]
    set_name = 'TestSet'
    save_type = 'pt'  # 'pt' or 'npz'
    execute_task(run_task_id, experiment_ids, scale_factor, padding_size, set_name, save_type)
