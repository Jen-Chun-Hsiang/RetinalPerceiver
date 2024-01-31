import os
import h5py
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor

def normalize_image(image_array):
    """Normalize image array to the range [-1, 1]."""
    return (image_array / 255.0) * 2.0 - 1.0

def process_session_images(session_path, output_file):
    """Process all images in a session and save them in an HDF5 file."""
    with h5py.File(output_file, "w") as hfile:
        # Iterate over all image files in the session directory
        for filename in os.listdir(session_path):
            if filename.endswith('.png'):
                frame_index = int(filename.split('_')[-1].split('.')[0])  # Extract frame index from filename
                img_path = os.path.join(session_path, filename)
                img = Image.open(img_path)
                img_array = np.array(img).astype(np.float32)

                # Normalize the image
                img_array = normalize_image(img_array)

                # Store the normalized image array in the HDF5 file with frame index as the key
                hfile.create_dataset(str(frame_index), data=img_array, compression="gzip")

def process_experiment_folders(root_folder):
    """Process all experiment folders in the root folder."""
    for folder in os.listdir(root_folder):
        experiment_path = os.path.join(root_folder, folder)
        if os.path.isdir(experiment_path) and folder.startswith('experiment_'):
            for subfolder in os.listdir(experiment_path):
                session_path = os.path.join(experiment_path, subfolder)
                if os.path.isdir(session_path) and subfolder.startswith('session_'):
                    output_file = os.path.join(experiment_path, f'{subfolder}.hdf5')
                    process_session_images(session_path, output_file)
                    print(f'Processed {subfolder} into {output_file}')


class PNGToTensorConverter:
    def __init__(self, root_dir, overwrite=False):
        self.root_dir = root_dir
        self.overwrite = overwrite
        self.to_tensor = ToTensor()

    def convert_directory(self, directory):
        for item in os.listdir(directory):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                # Recursively convert subdirectories
                self.convert_directory(path)
            elif path.endswith('.png'):
                self.convert_png_to_tensor(path)

    def convert_png_to_tensor(self, file_path):
        tensor_file_path = file_path.replace('.png', '.pt')

        # Check if tensor file already exists and overwrite flag
        if not self.overwrite and os.path.exists(tensor_file_path):
            print(f"Skipping {file_path} as tensor file already exists.")
            return

        # Load the image and convert it to a tensor
        image = Image.open(file_path)
        tensor = self.to_tensor(image)

        # Save the tensor
        torch.save(tensor, tensor_file_path)
        print(f"Saved tensor to {tensor_file_path}")

    def start_conversion(self):
        self.convert_directory(self.root_dir)


