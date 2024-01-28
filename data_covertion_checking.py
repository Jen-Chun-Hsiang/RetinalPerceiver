import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor


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


convert_root = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/'

# Usage
# To overwrite existing .pt files:
converter = PNGToTensorConverter(convert_root, overwrite=True)
converter.start_conversion()

'''
# To skip conversion if .pt file exists:
converter = PNGToTensorConverter('/path/to/your/folder', overwrite=False)
converter.start_conversion()
'''