import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import os
import time
import numpy as np


class ZarrDebugDataset(Dataset):
    """
    PyTorch Dataset backed by a Zarr array for debugging.
    """

    def __init__(self, zarr_path, start_time, pause_time=0.5):
        self.store = zarr.open(zarr_path, mode='r')
        self.data = self.store['data']
        self.start_time = start_time  # Store the initial perf_counter time
        self.pause_time = pause_time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        worker_id = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0
        elapsed_time = time.perf_counter() - self.start_time  # Calculate elapsed time
        print(f"[{elapsed_time:.2f}s] Worker {worker_id} processing index {index}")
        time.sleep(self.pause_time)  # Simulate workload
        return self.data[index]


def create_zarr_dataset(output_dir, dataset_size=100):
    """
    Create a Zarr dataset for testing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a Zarr group
    store = zarr.DirectoryStore(os.path.join(output_dir, 'debug_dataset.zarr'))
    root = zarr.group(store)

    # Create a dataset with random integers
    data = np.arange(dataset_size)
    root.create_dataset('data', data=data, chunks=(10,), dtype='int32')

    print(f"Zarr dataset created at {output_dir}/debug_dataset.zarr")


def test_dataloader(zarr_path, num_workers=0, batch_size=1, pause_time=0.5):
    """
    Test the DataLoader with the Zarr dataset.
    """
    start_time = time.perf_counter()  # Record the start time
    dataset = ZarrDebugDataset(zarr_path=zarr_path, start_time=start_time, pause_time=pause_time)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    print(f"Main process PID: {os.getpid()} - Starting DataLoader")

    try:
        for batch_idx, batch in enumerate(dataloader):
            elapsed_time = time.perf_counter() - start_time  # Calculate elapsed time
            print(f"[{elapsed_time:.2f}s] Main Process - Batch {batch_idx}: {batch.tolist()}")
    except Exception as e:
        print(f"Error during DataLoader operation: {e}")

    total_elapsed_time = time.perf_counter() - start_time  # Calculate total elapsed time
    print(f"Total time taken: {total_elapsed_time:.2f} seconds")