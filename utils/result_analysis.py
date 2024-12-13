import numpy as np
import torch
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
    result = pairwise_mult.sum(axis=(1, 2))

    return np.array(result)


class STAmodelEvaluator:
    """
    STAmodelEvaluator evaluates a model to compute Spike Triggered Average (STA) from given data.
    It now takes a DataLoader directly, allowing the user to benefit from multi-worker data loading.
    """

    def __init__(self, model, device, logger=None):
        """
        Initialize the STAmodelEvaluator with a model and device.

        Parameters
        ----------
        model : torch.nn.Module
            The trained model to evaluate.
        device : torch.device
            The device (CPU or GPU) for evaluation.
        logger : optional
            A logger instance for logging messages.
        """
        self.model = model
        self.device = device
        self.logger = logger

    def evaluate(self, dataloader, query_array=None, is_weight_in_label=False):

        self.model.eval()  # Model in evaluation mode
        query_array_tensor = torch.from_numpy(query_array).unsqueeze(1).float().to(self.device)
        # Prepare query array tensor if provided

        batch_size = dataloader.batch_size
        if batch_size is None:
            raise ValueError("DataLoader must have a defined batch_size.")

        # First pass: collect weights & labels
        all_weights, all_labels, batch_indices_tensor, within_batch_indices_tensor = self._collect_weights_and_labels(
            dataloader, query_array_tensor)

        if self.logger:
            self.logger.info('Finished collecting weights and labels.')

        # Compute normalized weights
        normalized_weights = self._compute_normalized_weights(all_weights, all_labels, is_weight_in_label)

        # Determine sample shape from dataset
        sample_shape = dataloader.dataset[0][0].shape
        weighted_sum = self._compute_weighted_sum(dataloader, normalized_weights)

        return weighted_sum, all_weights, all_labels

    def _collect_weights_and_labels(self, dataloader, query_array_tensor):
        """
        First pass through the dataset to collect weights and labels.
        """
        all_weights = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                images, labels, matrix_indices = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Get model weights for this batch
                weights, within_idx = self._get_weights_for_batch(images, query_array_tensor)

                if weights is not None:
                    weights_list = weights.cpu().tolist()
                    all_weights.extend(weights_list)
                else:
                    raise RuntimeError('weights are none!')
                    # If labels are weights, we just store labels below

                label_list = labels.cpu().tolist() if torch.is_tensor(labels) else labels
                all_labels.extend(label_list)

        return all_weights, all_labels

    def _get_weights_for_batch(self, images, query_array_tensor):
        """
        Compute model outputs (weights) for a single batch.
        """
        query_vectors = query_array_tensor.repeat(images.size(0), 1, 1)
        weights, _ = self.model(images, query_vectors)
        weights = weights.squeeze()
        within_idx = torch.arange(weights.size(0), device=self.device)

        return weights, within_idx

    def _compute_normalized_weights(self, all_weights, all_labels, is_weight_in_label):
        """Compute normalized weights."""
        if is_weight_in_label:
            weights_tensor = torch.tensor(all_labels)
        else:
            weights_tensor = torch.tensor(all_weights)

        weights_mean = weights_tensor.mean()
        weights_std = weights_tensor.std()
        normalized_weights = (weights_tensor - weights_mean) / (weights_std + 1e-7)
        return normalized_weights

    def _compute_weighted_sum(self, dataloader, normalized_weights, sample_shape):
        """
        Second pass to compute the STA by summing images weighted by normalized weights.
        """
        if len(sample_shape) == 4:
            C, D, H, W = sample_shape
            weighted_sum = torch.zeros((C, D, H, W), device=self.device)
        else:
            raise ValueError(f'Unexpected image dimensions {sample_shape}')

        idx = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                images, _, matrix_indices = data
                images = images.to(self.device)

                weights_batch = normalized_weights[idx:idx + images.size(0)].to(self.device).view(-1, 1, 1, 1, 1)

                # If no images remain after masking, skip
                if images.size(0) == 0:
                    continue

                weighted_images = images * weights_batch
                batch_sum = weighted_images.sum(dim=0)
                weighted_sum += batch_sum
                idx += images.size(0)

        return weighted_sum
