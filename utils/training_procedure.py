import torch
from torch.utils.data import DataLoader

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device):
    model.train()  # Set the model to training mode
    total_train_loss = 0

    for batch_idx, (input_matrices, targets) in enumerate(train_loader):
        input_matrices, targets = input_matrices.to(device), targets.to(device)

        # Adjusting shape for MSE loss, if needed
        targets = targets.unsqueeze(1)

        outputs = model(input_matrices)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss


def evaluate(test_loader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0

    with torch.no_grad():
        for input_matrices, targets in test_loader:
            input_matrices, targets = input_matrices.to(device), targets.to(device)

            # Adjust shape for MSE loss, if needed
            targets = targets.unsqueeze(1)

            outputs = model(input_matrices)
            loss = criterion(outputs, targets)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(test_loader)
    return avg_val_loss

def save_checkpoint(epoch, model, optimizer, training_losses, validation_losses, file_path):
    """
    Saves a checkpoint of the training process.

    Args:
    epoch (int): The current epoch number.
    model (torch.nn.Module): The model being trained.
    optimizer (torch.optim.Optimizer): The optimizer used for training.
    training_losses (list): List of training losses.
    validation_losses (list): List of validation losses.
    file_path (str): Path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_losses': training_losses,
        'validation_losses': validation_losses
    }
    torch.save(checkpoint, file_path)

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load a training checkpoint.

    Args:
    checkpoint_path (str): Path to the checkpoint file.
    model (torch.nn.Module): The model to load the state into.
    optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    device (torch.device): The device to map the model to.

    Returns:
    int: The epoch at which training was interrupted.
    model: The model with loaded state.
    optimizer: The optimizer with loaded state.
    list: The list of recorded training losses.
    list: The list of recorded validation losses.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    training_losses = checkpoint.get('training_losses', [])
    validation_losses = checkpoint.get('validation_losses', [])

    return start_epoch, model, optimizer, training_losses, validation_losses

def forward_model(model, dataset, batch_size=32):
    model.eval()  # Set the model to evaluation mode

    all_weights = []
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # First pass: Compute weights for all images
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(next(model.parameters()).device)
            weights = model(images).squeeze()

            all_weights.extend(weights.cpu().tolist())
            all_labels.extend(labels.cpu().tolist() if torch.is_tensor(labels) else labels)

    # Normalize weights
    weights_tensor = torch.tensor(all_weights)
    weights_mean = weights_tensor.mean()
    weights_std = weights_tensor.std()
    normalized_weights = (weights_tensor - weights_mean) / weights_std

    # Initialize weighted_sum before the loop
    sample_shape = dataloader.dataset[0][0].shape
    if len(sample_shape) == 4:  # Check if the sample has 4 dimensions
        C, D, H, W = sample_shape  # Assuming the shape is (C, D, H, W)
        weighted_sum = torch.zeros((C, D, H, W), device=next(model.parameters()).device)
    else:
        raise ValueError(f'Unexpected image dimensions {sample_shape}')

    idx = 0  # Index to track position in normalized weights
    for images, _ in dataloader:
        images = images.to(next(model.parameters()).device)
        weights_batch = normalized_weights[idx:idx + images.size(0)].to(images.device).view(-1, 1, 1, 1, 1)
        weighted_images = images * weights_batch
        batch_sum = weighted_images.sum(dim=0)  # Sum over the batch

        if weighted_sum is None:
            weighted_sum = batch_sum
        else:
            weighted_sum += batch_sum

        idx += images.size(0)

    return weighted_sum, all_weights, all_labels


