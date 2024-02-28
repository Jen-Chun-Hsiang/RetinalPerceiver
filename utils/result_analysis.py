import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, label, find_objects
from scipy.stats import mode

def find_connected_center(array_2d, connectivity=8):
    """
    Apply filtering and find the center of mass of the largest connected component
    in terms of peak intensity.

    Args:
    - array_2d: 2D numpy array.
    - connectivity: Connectivity for connected component analysis; 8 for 8-connectivity.

    Returns:
    - A tuple (y_center, x_center) as the center of mass of the selected connected component.
    """
    # Apply a 2D median filter with a 3x3 kernel
    median_filtered = median_filter(array_2d, size=3)

    # Apply a Gaussian filter with a 5x5 kernel
    gaussian_filtered = gaussian_filter(median_filtered, sigma=5 / 6)

    # Threshold the filtered image to get the most significant features
    threshold = np.mean(gaussian_filtered) + np.std(gaussian_filtered)
    binary_image = gaussian_filtered > threshold

    # Label connected components
    labeled_array, num_features = label(binary_image, structure=np.ones((3, 3)))

    # Find the connected component with the highest peak
    peak_intensities = [np.max(gaussian_filtered[labeled_array == i]) for i in range(1, num_features + 1)]
    if not peak_intensities:
        return None  # Return None if no components are found

    highest_peak_component = np.argmax(peak_intensities) + 1  # Component labels start at 1

    # Find the center of mass of the component with the highest peak
    component_slice = find_objects(labeled_array == highest_peak_component)[0]
    component = gaussian_filtered[component_slice]
    y, x = np.indices(component.shape)

    total_intensity = component.sum()
    x_center = (x * component).sum() / total_intensity
    y_center = (y * component).sum() / total_intensity

    # Adjust the center coordinates relative to the original image
    y_center += component_slice[0].start
    x_center += component_slice[1].start

    return np.array((y_center, x_center))


def pairwise_mult_sum(matrix_2d, array_3d):
    """
    Perform pairwise multiplication between a 2D array and each slice of a 3D array along the last dimension,
    and then sum over the first two dimensions to result in a 1D array of shape [1 x D].

    Args:
    - matrix_2d: A 2D numpy array of shape (N, M).
    - array_3d: A 3D numpy array of shape (N, M, D).

    Returns:
    - A 1D numpy array of shape [1 x D] resulting from the pairwise multiplication and summation.
    """
    # Check the shapes to ensure compatibility for pairwise multiplication
    if matrix_2d.shape != array_3d.shape[:2]:
        raise ValueError("The first two dimensions of matrix_2d and array_3d must match.")

    # Perform pairwise multiplication
    # The result of this operation is a 3D array with the same shape as array_3d
    pairwise_mult = matrix_2d[:, :, np.newaxis] * array_3d

    # Sum over the first two dimensions to collapse them
    # The resulting shape will be [1 x D]
    result = pairwise_mult.sum(axis=(0, 1))

    return np.array(result)
