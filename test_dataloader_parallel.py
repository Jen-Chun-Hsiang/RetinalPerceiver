import torch
from torch.utils.data import Dataset, DataLoader
import os

from utils.parallel_test import create_zarr_dataset, test_dataloader


if __name__ == "__main__":
    mode = 3
    output_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Debugs/'
    dataset_size = 100
    num_workers = 3
    batch_size = 5
    pause_time = 0.2

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zarr_path = os.path.join(output_dir, 'debug_dataset.zarr')

    if mode == 1:
        print("Running in mode 1: Generate Data Only")
        create_zarr_dataset(output_dir, dataset_size)

    elif mode == 2:
        print("Running in mode 2: Test DataLoader Only")
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr dataset not found at {zarr_path}. Please generate the dataset first.")
        test_dataloader(zarr_path, num_workers=num_workers, batch_size=batch_size, pause_time=pause_time)

    elif mode == 3:
        print("Running in mode 3: Combine Data Generation and DataLoader Testing")
        create_zarr_dataset(output_dir, dataset_size)
        test_dataloader(zarr_path, num_workers=num_workers, batch_size=batch_size, pause_time=pause_time)
