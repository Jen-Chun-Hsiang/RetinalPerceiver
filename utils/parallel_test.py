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


def create_zarr_dataset(output_dir, dataset_gb=1, chunk_size=10000):
    """
    Efficiently create a Zarr dataset that expands to the specified memory size.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate the number of elements needed to reach the specified size
    element_size_bytes = np.dtype('int32').itemsize  # Each int32 is 4 bytes
    total_elements = (dataset_gb * 1024 ** 3) // element_size_bytes

    # Create a Zarr group
    store = zarr.DirectoryStore(os.path.join(output_dir, 'debug_dataset.zarr'))
    root = zarr.group(store)

    if 'data' in root:
        print("Dataset 'data' already exists. Overwriting...")
        del root['data']

    # Create a dataset with a specified chunk size for efficient access
    data = root.create_dataset(
        'data',
        shape=(total_elements,),
        chunks=(chunk_size,),
        dtype='int32',
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),  # Compression for efficiency
        overwrite=True  # Allow overwriting of the existing dataset
    )

    print(f"Creating Zarr dataset with {total_elements} elements (~{dataset_gb} GB) in chunks of {chunk_size}...")

    # Use a generator to populate the dataset in chunks efficiently
    def data_generator():
        for i in range(0, total_elements, chunk_size):
            end = min(i + chunk_size, total_elements)
            yield np.arange(i, end, dtype='int32')

    # Write data in parallel using Zarr's array parallel writes
    from concurrent.futures import ThreadPoolExecutor

    def write_chunk(chunk_index, chunk_data):
        start_idx = chunk_index * chunk_size
        end_idx = start_idx + len(chunk_data)
        data[start_idx:end_idx] = chunk_data
        print(f"Chunk {chunk_index} written: [{start_idx}:{end_idx}]")

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(write_chunk, i, chunk_data)
            for i, chunk_data in enumerate(data_generator())
        ]
        for future in futures:
            future.result()  # Ensure all tasks are completed

    print(f"Zarr dataset created at {output_dir}/debug_dataset.zarr, size: {dataset_gb} GB")


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