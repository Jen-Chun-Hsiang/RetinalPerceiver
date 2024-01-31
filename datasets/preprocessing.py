import os
import h5py
import numpy as np
from PIL import Image

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

'''
# Set the root folder path
root_folder = 'path/to/root/folder'
process_experiment_folders(root_folder)
'''