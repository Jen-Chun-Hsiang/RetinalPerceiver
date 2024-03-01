import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, label, find_objects


def find_connected_center(array_2d, connectivity=8, top_n=10):
    """
    Find the combined center of mass of the top N connected components based on peak intensity.

    Args:
    - array_2d: 2D numpy array.
    - connectivity: Connectivity for connected component analysis; 8 for 8-connectivity.
    - top_n: Number of top connected components to consider.

    Returns:
    - A tuple (y_center, x_center) as the combined center of mass of the top connected components.
    """
    # Apply a 2D median filter with a 3x3 kernel
    median_filtered = median_filter(array_2d, size=3)

    # Apply a Gaussian filter with a 5x5 kernel
    gaussian_filtered = gaussian_filter(median_filtered, sigma=5 / 6)

    # Threshold the filtered image to get significant features
    threshold = np.mean(gaussian_filtered) + np.std(gaussian_filtered)
    binary_image = gaussian_filtered > threshold

    # Label connected components
    structure = np.ones((3, 3)) if connectivity == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled_array, num_features = label(binary_image, structure=structure)

    # Find the peak intensity of each component
    peak_intensities = [np.max(gaussian_filtered[labeled_array == i]) for i in range(1, num_features + 1)]

    # Sort components by peak intensity and select the top N
    top_components_indices = np.argsort(peak_intensities)[::-1][:top_n]  # Get indices of top N components

    # Create a combined mask for the top N components
    combined_mask = np.isin(labeled_array, top_components_indices + 1)  # Adjust for 1-based indexing of components

    # Calculate the weighted coordinates for the center of mass
    y_indices, x_indices = np.indices(array_2d.shape)
    total_intensity = np.sum(gaussian_filtered[combined_mask])
    x_center = np.sum(x_indices[combined_mask] * gaussian_filtered[combined_mask]) / total_intensity
    y_center = np.sum(y_indices[combined_mask] * gaussian_filtered[combined_mask]) / total_intensity

    return np.array((y_center, x_center))


def pairwise_mult_sum(matrix_2d, array_3d):
    """
    Perform pairwise multiplication between a 2D array and each slice of a 3D array along the last dimension,
    and then sum over the first two dimensions to result in a 1D array of shape [1 x D].

    Args:
    - matrix_2d: A 2D numpy array of shape (H, W).
    - array_3d: A 3D numpy array of shape (D, H, W).

    Returns:
    - A 1D numpy array of shape [1 x D] resulting from the pairwise multiplication and summation.
    """
    # Check the shapes to ensure compatibility for pairwise multiplication
    if matrix_2d.shape != array_3d.shape[1:3]:
        raise ValueError("The first two dimensions of matrix_2d and array_3d must match.")

    # Perform pairwise multiplication
    # The result of this operation is a 3D array with the same shape as array_3d
    pairwise_mult = matrix_2d[np.newaxis, :, :] * array_3d

    # Sum over the first two dimensions to collapse them
    # The resulting shape will be [1 x D]
    result = pairwise_mult.sum(axis=(0, 1))

    return np.array(result)
