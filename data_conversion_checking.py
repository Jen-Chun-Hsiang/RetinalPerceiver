from datasets.preprocessing import PNGToTensorConverter
from datasets.preprocessing import process_experiment_folders

task_id = 1


def convert_png_to_ph_overwrite():
    # (1) Convert png to ph
    convert_root = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/experiment_1/'
    converter = PNGToTensorConverter(convert_root, overwrite=True)
    converter.start_conversion()


def convert_png_to_ph_addnew():
    # (2) Skip conversion if .pt file exists:
    convert_root = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/experiment_1/'
    converter = PNGToTensorConverter(convert_root, overwrite=False)
    converter.start_conversion()


def generate_session_in_hdf5():
    # (3) Gather image into hdf5 file
    root_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/experiment_1/'
    process_experiment_folders(root_folder)


def execute_task(argument):
    switcher = {
        1: convert_png_to_ph_overwrite,
        2: convert_png_to_ph_addnew,
        3: generate_session_in_hdf5,
    }
    # Attempt to retrieve the function from the switcher dictionary
    func = switcher.get(argument)

    if func is None:
        raise ValueError("No task found for argument: {}".format(argument))

    # Execute the function and return its result
    return func()


execute_task(task_id)