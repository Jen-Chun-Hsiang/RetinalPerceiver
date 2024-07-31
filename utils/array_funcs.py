import numpy as np


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