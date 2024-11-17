import torch
from torch.utils.data import Dataset, DataLoader
import os
import time


class DebugDataset(Dataset):
    """Dataset with simulated workload and logging for debugging."""

    def __init__(self, size, start_time):
        self.data = list(range(size))
        self.start_time = start_time  # Store the initial perf_counter time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        worker_id = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0
        elapsed_time = time.perf_counter() - self.start_time  # Calculate elapsed time
        print(f"[{elapsed_time:.2f}s] Worker {worker_id} processing index {index}")
        time.sleep(0.5)  # Simulate workload
        return self.data[index]


def test_dataloader(num_workers, num_samples, batch_size):
    start_time = time.perf_counter()  # Record the start time
    dataset = DebugDataset(size=num_samples, start_time=start_time)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    print(f"Main process PID: {os.getpid()} - Starting DataLoader")

    try:
        for batch_idx, batch in enumerate(dataloader):
            elapsed_time = time.perf_counter() - start_time  # Calculate elapsed time
            print(f"[{elapsed_time:.2f}s] Main Process - Batch {batch_idx}: {batch}")
    except Exception as e:
        print(f"Error during DataLoader operation: {e}")

    total_elapsed_time = time.perf_counter() - start_time  # Calculate total elapsed time
    print(f"Total time taken: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)  # Use spawn method for compatibility
    num_workers = 3  # Test with more than 1 worker
    num_samples = 100
    batch_size = 5
    test_dataloader(num_workers, num_samples, batch_size)
