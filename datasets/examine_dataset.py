import torch
from torch.utils.data import DataLoader


def STA_check(dataset, batch_size=16, device='cuda'):
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for data in dataloader:
        images, labels, matrix_indices = data
        all_labels.extend(labels.cpu().tolist() if torch.is_tensor(labels) else labels)
    '''
    weights_tensor = torch.tensor(all_labels)
    weights_mean = weights_tensor.mean()
    weights_std = weights_tensor.std()
    normalized_weights = (weights_tensor - weights_mean) / weights_std
    '''
    weights_tensor = torch.tensor(all_labels)
    weights_sum = weights_tensor.sum()
    normalized_weights = weights_tensor / weights_sum

    # Initialize weighted_sum before the loop
    sample_shape = dataloader.dataset[0][0].shape
    if len(sample_shape) == 4:  # Check if the sample has 4 dimensions
        C, D, H, W = sample_shape  # Assuming the shape is (C, D, H, W)
        weighted_sum = torch.zeros((C, D, H, W), device=device)
    else:
        raise ValueError(f'Unexpected image dimensions {sample_shape}')

    idx = 0  # Index to track position in normalized weights
    for data in dataloader:
        images, _, matrix_indices = data
        # query_vectors = query_array_tensor[matrix_indices].to(images.device)
        weights_batch = normalized_weights[idx:idx + images.size(0)].to(images.device).view(-1, 1, 1, 1, 1)
        weighted_images = images * weights_batch

        batch_sum = weighted_images.sum(dim=0)  # Sum over the batch

        if weighted_sum is None:
            weighted_sum = batch_sum
        else:
            weighted_sum += batch_sum

        idx += images.size(0)

    return weighted_sum
