import numpy as np
import os
import zarr
import numcodecs


def update_unique_array(B, D):
    """
    Update matrix B by appending unique rows from matrix D that are not already in B.

    Args:
    B (np.array): The original matrix B with unique rows.
    D (np.array): New input matrix D with the same number of columns as B.

    Returns:
    np.array: Updated matrix B with unique rows from D appended.
    """
    if B.shape[1] != D.shape[1]:
        raise ValueError("Both matrices must have the same number of columns")

    # Convert rows to tuples for set operations
    existing_rows = set(map(tuple, B))
    new_rows = set(map(tuple, D))

    # Find rows in D not already in B
    unique_new_rows = new_rows - existing_rows

    # If there are new unique rows, append them to B
    if unique_new_rows:
        unique_new_rows = np.array(list(unique_new_rows))
        B = np.vstack((B, unique_new_rows))

    return B


def find_matching_indices_in_arrays(A, B):
    """
    Find indices C such that B[C] equals A, where A can have repeated rows and B is unique.

    Args:
    A (np.array): Matrix A of size m x n.
    B (np.array): Matrix B of size p x n, where p <= m and B has unique rows.

    Returns:
    np.array: An array of indices C such that B[C] equals A.
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError("Both matrices must have the same number of columns")

    # Create a dictionary to map rows of B to their indices for quick lookup
    row_to_index = {tuple(row): idx for idx, row in enumerate(B)}

    C = []
    for row in A:
        row_tuple = tuple(row)
        if row_tuple in row_to_index:
            C.append(row_to_index[row_tuple])
        else:
            raise ValueError(f"No matching row in B for row {row} in A")

    return np.array(C)


def split_array(arr, num_sets):
    # Calculate the size of each set
    set_size = len(arr) // num_sets

    # Shuffle the array indices to ensure randomness
    shuffled_indices = np.random.permutation(len(arr))

    # Create the sets using the shuffled indices
    equal_sets = [arr[shuffled_indices[i * set_size:(i + 1) * set_size]] for i in range(num_sets)]

    return equal_sets


def load_keyword_based_arrays(file_folder, keyword, dtype=np.int32):
    # Create an empty list to store the memory-mapped arrays
    arrays = []

    # Initialize the index to find files sequentially starting from 0
    i = 0
    while True:
        # Construct the file name based on the keyword and index
        file_name = f'{keyword}_{i}.npy'
        file_path = os.path.join(file_folder, file_name)

        # Check if the file exists
        if os.path.isfile(file_path):
            # Load the file as a memory-mapped array
            # Use the provided dtype for the data type of the numpy array
            # Load the shape using np.load to determine the shape for memmap
            array = np.memmap(file_path, dtype=dtype, mode='r', shape=np.load(file_path, mmap_mode='r').shape)
            #z_saved = zarr.open(session_data_path, mode='w', shape=session_data.shape, dtype=object,
            #                    object_codec=numcodecs.Pickle())
            #z_saved[:] = session_data
            # Display the shape of the memmap array
            print("Shape of the memmap array:", array.shape)

            # Print the first 10 rows and first 5 columns
            print("First 10 rows and first 5 columns of the array:")
            print(array[:10, :5])

            arrays.append(array)

            i += 1  # Move to the next file index
        else:
            break  # Exit the loop if the file does not exist

    return arrays


class VirtualArraySampler:
    def __init__(self, arrays):
        # Convert 1D arrays to 2D arrays with one column each
        self.arrays = [a[:, np.newaxis] if a.ndim == 1 else a for a in arrays]

        '''
        # Assuming `arrays` is a list of numpy arrays or similar array-like structures
        for i, a_check in enumerate(self.arrays):
            print(f"Array {i}:")
            try:
                shape = a_check.shape
                print(f"Shape: {shape}")
                # Check if the array is empty by verifying if any dimension is zero
                if a_check.size == 0:
                    print("Status: Empty array")
                else:
                    print("Status: Non-empty array")
            except AttributeError:
                print("This item does not have a shape or size attribute. It might not be an array.")
            print("\n")  # Adds a newline for better readability between arrays
        '''

        # Ensure all arrays have the same number of columns
        if not all(a.shape[1] == self.arrays[0].shape[1] for a in self.arrays):
            raise ValueError("All arrays must have the same number of columns.")

        self.shapes = [a.shape[0] for a in self.arrays]
        self.start_indices = np.cumsum([0] + self.shapes[:-1])
        self.end_indices = np.cumsum(self.shapes) - 1
        self.total_length = sum(self.shapes)
        self.num_columns = self.arrays[0].shape[1]

    def total_rows(self):
        return self.total_length

    def total_columns(self):
        return self.num_columns

    def sample(self, indices):
        # Ensure indices are within bounds
        if np.any(indices >= self.total_length) or np.any(indices < 0):
            raise ValueError("Indices are out of bounds.")

        # Find out which array each index belongs to
        array_indices = np.searchsorted(self.start_indices, indices, side='right') - 1

        # Prepare to gather results
        results = np.empty((len(indices), self.arrays[0].shape[1]), dtype=self.arrays[0].dtype)

        # Retrieve all necessary rows from each array in one operation
        for array_idx in range(len(self.arrays)):
            # Find indices belonging to the current array
            idx_in_array = indices[array_indices == array_idx] - self.start_indices[array_idx]
            if len(idx_in_array) > 0:
                results[array_indices == array_idx] = self.arrays[array_idx][idx_in_array]

        return results


class ZarrSampler:
    def __init__(self, zarr_path, chunk_size):
        self.zarr_array = zarr.open(zarr_path, mode='r')
        self.chunk_size = chunk_size
        self.total_length = self.zarr_array.shape[0]
        self.num_columns = self.zarr_array.shape[1] if len(self.zarr_array.shape) > 1 else 1
        self.start_indices = np.arange(0, self.total_length, self.chunk_size)

        # Automatically print the first few rows upon initialization
        # self.print_first_check()

    def print_first_check(self, n_rows=10, n_cols=5):
        """Prints the first n rows of the zarr array."""
        # Check if the array has enough rows
        if self.total_length < n_rows:
            n_rows = self.total_length  # Adjust if total_length is less than requested rows
        if self.num_columns < n_cols:
            n_cols = self.num_columns  # Adjust if total_length is less than requested rows
        first_rows = self.zarr_array[273564:273564+n_rows, :n_cols]  # Slice the first n rows
        print(f'total length: {self.total_length}')
        print("First few rows of the Zarr array:")
        print(first_rows)

    def sample(self, indices):
        # Ensure indices are within bounds
        if np.any(indices >= self.total_length) or np.any(indices < 0):
            raise ValueError("Indices are out of bounds.")

        # Find out which array each index belongs to
        array_indices = np.searchsorted(self.start_indices, indices, side='right') - 1

        # Prepare to gather results
        results = np.empty((len(indices), self.num_columns), dtype=self.zarr_array.dtype)

        # Retrieve all necessary rows from each array in one operation
        for array_idx in np.unique(array_indices):
            # Find indices belonging to the current array
            idx_in_array = indices[array_indices == array_idx] - self.start_indices[array_idx]
            if idx_in_array.size > 0:
                results[array_indices == array_idx] = self.zarr_array[array_idx * self.chunk_size:(array_idx + 1) * self.chunk_size][idx_in_array]

        return results

    def total_rows(self):
        return self.total_length

    def total_columns(self):
        return self.num_columns


def calculate_num_sets(num_rows, num_columns, dtype, max_array_bank_capacity=1e9):
    """
    Calculate the number of sets needed to split a large 2D numpy array into smaller ones,
    each with a size not exceeding max_array_bank_capacity bytes.

    Parameters:
    - num_rows: int, number of rows in the original 2D array
    - num_columns: int, number of columns in the original 2D array
    - dtype: data type of the numpy array (e.g., np.int32, np.float64)
    - max_array_bank_capacity: float, maximum capacity of each array bank in bytes (default is 1GB)

    Returns:
    - num_sets: int, the number of sets required
    """
    # Determine the size of one element based on the dtype
    element_size = np.dtype(dtype).itemsize

    # Calculate the size of the full 2D array
    total_array_size = num_rows * num_columns * element_size

    # Calculate the number of sets required
    num_sets = np.ceil(total_array_size / max_array_bank_capacity)

    return int(num_sets)