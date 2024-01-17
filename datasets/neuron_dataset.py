import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import random

def precompute_image_paths(data_array, root_dir):
    precomputed_paths = {}
    for row in data_array:
        experiment_idx = row[0]
        session_idx = row[1]
        image_frame_indices = row[2:]
        for image_frame_idx in image_frame_indices:
            key = (experiment_idx, session_idx, image_frame_idx)
            path = os.path.join(root_dir,
                                f"experiment_{experiment_idx}",
                                f"session_{session_idx}",
                                f"frame_{image_frame_idx}.png")
            precomputed_paths[key] = path
    return precomputed_paths

class RetinalDataset(Dataset):
    def __init__(self, data_array, query_series, image_root_dir, chunk_indices, chunk_size, device='cuda', use_cache=True):
        self.data_array = data_array
        self.query_series = query_series
        self.image_root_dir = image_root_dir
        self.chunk_indices = chunk_indices
        self.chunk_size = chunk_size
        self.device = device
        self.use_cache = use_cache
        self.image_paths = precompute_image_paths(data_array, image_root_dir)
        self.image_tensor_cache = {}
        assert len(data_array) == len(query_series), "data_array and query_series must be the same length"

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        # Determine the range of the chunk
        start_idx = self.chunk_indices[idx]
        end_idx = min(start_idx + self.chunk_size, len(self.data_array))

        # Randomly select a data point within the chunk
        random_idx = random.randint(start_idx, end_idx - 1)
        firing_rate, experiment_id, session_id, neuron_id, *frame_ids = self.data_array[random_idx]
        query_id = self.query_series[random_idx]

        # Load and stack images for the selected data point
        images = [self.load_image(experiment_id, session_id, frame_id) for frame_id in frame_ids]
        images_3d = torch.stack(images, dim=0)

        return images_3d, firing_rate, query_id

    def load_image(self, experiment_id, session_id, frame_id):
        key = (experiment_id, session_id, frame_id)
        if self.use_cache and key in self.image_tensor_cache:
            return self.image_tensor_cache[key]
        image_path = self.image_paths[key]
        image = Image.open(image_path)
        image_tensor = torch.from_numpy(np.array(image)).float().to(self.device)
        if self.use_cache:
            self.image_tensor_cache[key] = image_tensor
        return image_tensor

def train_val_split(data_length, chunk_size, test_size=0.2):
    total_chunks = (data_length - 1) // chunk_size + 1
    indices = np.arange(0, chunk_size * total_chunks, chunk_size)[:total_chunks]
    val_size = int(total_chunks * test_size)
    val_indices = np.random.choice(indices, size=val_size, replace=False)
    train_indices = np.setdiff1d(indices, val_indices)
    return train_indices, val_indices
