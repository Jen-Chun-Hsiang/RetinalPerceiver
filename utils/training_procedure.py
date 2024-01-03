import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, train_loader, epoch, query_array=None):
        self.model.train()  # Set the model to training mode
        total_train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            if query_array is not None:
                # Process with query vectors
                loss = self._process_batch_with_query(data, query_array)
            else:
                # Process as originally
                loss = self._process_batch(data)

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        return avg_train_loss

    def _process_batch(self, data):
        input_matrices, targets = data
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        targets = targets.unsqueeze(1)
        outputs = self.model(input_matrices)
        loss = self._compute_loss(outputs, targets)
        self._update_parameters(loss)
        return loss

    def _process_batch_with_query(self, data, query_array):
        input_matrices, targets, matrix_indices = data
        query_vectors = torch.from_numpy(query_array).unsqueeze(1)
        query_vectors = query_vectors[matrix_indices]
        query_vectors = query_vectors.float().to(self.device)
        #query_vectors = torch.ones(query_vectors.shape).to(self.device)
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        targets = targets.unsqueeze(1)
        outputs = self.model(input_matrices, query_vectors)
        loss = self._compute_loss(outputs, targets)
        self._update_parameters(loss)
        return loss

    def _compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def _update_parameters(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Evaluator:
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def evaluate(self, test_loader, query_array=None):
        self.model.eval()  # Set the model to evaluation mode
        total_val_loss = 0

        with torch.no_grad():
            for data in test_loader:
                if query_array is not None:
                    # Process with query vectors
                    loss = self._process_batch_with_query(data, query_array)
                else:
                    # Process as originally
                    loss = self._process_batch(data)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        return avg_val_loss

    def _process_batch(self, data):
        input_matrices, targets = data
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        targets = targets.unsqueeze(1)
        outputs = self.model(input_matrices)
        loss = self._compute_loss(outputs, targets)
        return loss

    def _process_batch_with_query(self, data, query_array):
        input_matrices, targets, matrix_indices = data
        query_vectors = torch.from_numpy(query_array).unsqueeze(1)
        query_vectors = query_vectors[matrix_indices]
        query_vectors = query_vectors.float().to(self.device)
        #query_vectors = torch.ones(query_vectors.shape).to(self.device)
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        targets = targets.unsqueeze(1)
        outputs = self.model(input_matrices, query_vectors)
        loss = self._compute_loss(outputs, targets)
        return loss

    def _compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)


def save_checkpoint(epoch, model, optimizer, args, training_losses, validation_losses, file_path):
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
        'args': args,
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

def forward_model(model, dataset, query_array=None, batch_size=32):
    model.eval()  # Set the model to evaluation mode

    all_weights = []
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare query array if provided
    query_array_tensor = torch.from_numpy(query_array).unsqueeze(1).float() if query_array is not None else None
    use_query = query_array_tensor is not None

    # First pass: Compute weights for all images
    with torch.no_grad():
        for data in dataloader:
            if use_query:
                images, labels, matrix_indices = data
                query_vectors = query_array_tensor[matrix_indices].to(images.device)
                weights = model(images, query_vectors).squeeze()
            else:
                images, labels = data
                weights = model(images).squeeze()

            #images = images.to(next(model.parameters()).device)
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
    for data in dataloader:
        if use_query:
            images, _, matrix_indices = data
            query_vectors = query_array_tensor[matrix_indices].to(images.device)
            weights_batch = normalized_weights[idx:idx + images.size(0)].to(images.device).view(-1, 1, 1, 1, 1)
            weighted_images = images * weights_batch
        else:
            images, _ = data
            images = images.to(next(model.parameters()).device)
            weights_batch = normalized_weights[idx:idx + images.size(0)].to(images.device).view(-1, 1, 1, 1)
            weighted_images = images * weights_batch

        batch_sum = weighted_images.sum(dim=0)  # Sum over the batch

        if weighted_sum is None:
            weighted_sum = batch_sum
        else:
            weighted_sum += batch_sum

        idx += images.size(0)

    return weighted_sum, all_weights, all_labels



