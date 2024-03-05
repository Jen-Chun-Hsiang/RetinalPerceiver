import torch
from torch.utils.data import DataLoader
from .loss_function import CosineNegativePairLoss
import numpy as np
from operator import itemgetter


class Trainer:
    def __init__(self, model, criterion, optimizer, device, accumulation_steps=1,
                 query_array=None, is_contrastive_learning=False,
                 query_encoder=None, query_permutator=None, series_ids=None,
                 margin=0.1, temperature=0.1):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.accumulation_steps = accumulation_steps
        if query_array is not None:
            self.is_query_array = True
            self.query_array = torch.from_numpy(query_array).unsqueeze(1)
        else:
            self.is_query_array = False
        self.is_contrastive_learning = is_contrastive_learning
        if self.is_contrastive_learning:
            if query_encoder is None or query_permutator is None or series_ids is None:
                raise ValueError("Not enough inputs in Trainer to perform contrastive learning.")

            self.query_encoder = query_encoder
            self.query_permutator = query_permutator
            self.series_ids = series_ids
            self.neg_contra_loss_fn = CosineNegativePairLoss(margin=margin, temperature=temperature)

    def train_one_epoch(self, train_loader):
        self.model.train()  # Set the model to training mode
        total_train_loss = 0
        self.optimizer.zero_grad()  # Zero gradients at the start of the epoch

        if len(train_loader) == 0:
            raise ValueError("train_loader is empty. The training process requires a non-empty train_loader.")

        for batch_idx, data in enumerate(train_loader):
            if self.is_query_array:
                if self.is_contrastive_learning:
                    loss = self._process_batch_with_query_contrast(data)
                else:
                    loss = self._process_batch_with_query(data)
            else:
                loss = self._process_batch(data)

            if loss is not None:
                self._update_parameters(loss)
                total_train_loss += loss.item()
            else:
                raise ValueError(f"Loss is None for batch {batch_idx}. Check your model's output and loss function.")

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()  # Perform optimization step
                self.optimizer.zero_grad()  # Zero gradients for the next accumulation

        avg_train_loss = total_train_loss / len(train_loader)
        return avg_train_loss

    def _process_batch(self, data):
        input_matrices, targets = data
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        targets = targets.unsqueeze(1)
        outputs = self.model(input_matrices)
        loss = self._compute_loss(outputs, targets)
        return loss

    def _process_batch_with_query(self, data):
        input_matrices, targets, matrix_indices = data
        query_vectors = self.query_array[matrix_indices]
        query_vectors = query_vectors.float().to(self.device)
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        outputs = self.model(input_matrices, query_vectors)
        try:
            assert outputs.shape == targets.shape
        except Exception as e:
            print(e)
            print(f'outputs shape: {outputs.shape}')
            print(f'targets shape: {targets.shape}')

        loss = self._compute_loss(outputs, targets)

        return loss

    def _process_batch_with_query_contrast(self, data):
        input_matrices, targets, matrix_indices = data
        query_vectors = self.query_array[matrix_indices]
        query_vectors = query_vectors.float().to(self.device)
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        outputs_predict, outputs_embedding = self.model(input_matrices, query_vectors)
        num_batch = input_matrices.shape[0]
        perm_series_ids = [self.series_ids[i] for i in matrix_indices]
        batch_permuted_queries = self.query_permutator.generate_batch_perm_list(perm_series_ids)
        contra_loss = 0
        for i, permuted_queries in enumerate(batch_permuted_queries):
            query_array = self.query_encoder.encode(permuted_queries)
            query_vectors = torch.from_numpy(query_array).unsqueeze(1)
            query_vectors = query_vectors.float().to(self.device)
            _, perm_embedding = self.model(input_matrices, query_vectors)
            contra_loss += self.neg_contra_loss_fn(perm_embedding.view(num_batch, -1),
                                                   outputs_embedding.view(num_batch, -1))

        return self._compute_loss(outputs_predict, targets) + contra_loss

    def _compute_loss(self, outputs, targets):
        return self.criterion(outputs.squeeze(), targets.squeeze())

    def _update_parameters(self, loss):
        loss.backward()  # Compute the backward pass only


class Evaluator(Trainer):
    def __init__(self, model, criterion, device,
                 query_array=None, is_contrastive_learning=False,
                 query_encoder=None, query_permutator=None, series_ids=None,
                 margin=0.1, temperature=0.1):
        # Initialize the parent class without an optimizer as it's not needed for evaluation
        super().__init__(model, criterion, None, device, query_array=query_array,
                         is_contrastive_learning=is_contrastive_learning, query_encoder=query_encoder,
                         series_ids=series_ids, query_permutator=query_permutator,
                         margin=margin, temperature=temperature)

    def evaluate(self, eval_loader):
        self.model.eval()  # Set the model to evaluation mode
        total_eval_loss = 0
        with torch.no_grad():  # Ensure no gradients are computed
            for batch_idx, data in enumerate(eval_loader):
                if self.is_query_array:
                    if self.is_contrastive_learning:
                        loss = self._process_batch_with_query_contrast(data)
                    else:
                        loss = self._process_batch_with_query(data)
                else:
                    loss = self._process_batch(data)

                if loss is not None:
                    total_eval_loss += loss.item()
                else:
                    raise ValueError(
                        f"Loss is None for batch {batch_idx}. Check your model's output and loss function.")

        avg_eval_loss = total_eval_loss / len(eval_loader)
        return avg_eval_loss

    def _update_parameters(self, loss):
        pass  # Override to prevent any parameter updates


'''
class Evaluator:
    def __init__(self, model, criterion, device, query_array=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        if query_array is not None:
            self.is_query_array = True
            self.query_array = torch.from_numpy(query_array).unsqueeze(1)
        else:
            self.is_query_array = False

    def evaluate(self, test_loader):
        self.model.eval()  # Set the model to evaluation mode
        total_val_loss = 0

        if len(test_loader) == 0:
            raise ValueError("test_loader is empty. The training process requires a non-empty train_loader.")

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                if self.is_query_array:
                    # Process with query vectors
                    loss = self._process_batch_with_query(data)
                else:
                    # Process as originally
                    loss = self._process_batch(data)

                if loss is not None:
                    total_val_loss += loss.item()
                else:
                    raise ValueError(
                        f"Loss is None for batch {batch_idx}. Check your model's output and loss function.")

        avg_val_loss = total_val_loss / len(test_loader)
        return avg_val_loss

    def _process_batch(self, data):
        input_matrices, targets = data
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        targets = targets.unsqueeze(1)
        outputs = self.model(input_matrices)
        loss = self._compute_loss(outputs, targets)
        return loss

    def _process_batch_with_query(self, data):
        input_matrices, targets, matrix_indices = data
        query_vectors = self.query_array[matrix_indices]
        query_vectors = query_vectors.float().to(self.device)
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        outputs = self.model(input_matrices, query_vectors)
        loss = self._compute_loss(outputs, targets)
        return loss

    def _compute_loss(self, outputs, targets):
        return self.criterion(outputs.squeeze(), targets.squeeze())
'''


def save_checkpoint(epoch, model, optimizer, args, training_losses,
                    validation_losses, validation_contra_losses, file_path):
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
        'validation_losses': validation_losses,
        'validation_contra_losses': validation_contra_losses
    }
    torch.save(checkpoint, file_path)


class CheckpointLoader:
    def __init__(self, checkpoint_path, device):
        self.checkpoint = None
        self.start_epoch = None
        self.training_losses = None
        self.validation_losses = None
        self.validation_contra_losses = None
        self.args = None
        self.checkpoint = torch.load(checkpoint_path, map_location=device)

    def load_args(self):
        """
        Load and return the 'args' from the checkpoint.

        Returns:
        dict: The 'args' used to create the model and optimizer.
        """

        self.args = self.checkpoint['args']
        return self.args

    def load_checkpoint(self, model, optimizer):
        """
        Load a training checkpoint into the model and optimizer.

        Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        device (torch.device): The device to map the model to.
        """
        model.load_state_dict(self.checkpoint['model_state_dict'])
        optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        return model, optimizer

    def load_epoch(self):
        """ Return the epoch at which training was interrupted. """
        self.start_epoch = self.checkpoint['epoch']
        return self.start_epoch

    def load_training_losses(self):
        """ Return the list of recorded training losses. """
        self.training_losses = self.checkpoint.get('training_losses', [])
        return self.training_losses

    def load_validation_losses(self):
        """ Return the list of recorded validation losses. """
        self.validation_losses = self.checkpoint.get('validation_losses', [])
        return self.validation_losses

    def load_validation_losses(self):
        """ Return the list of recorded validation losses. """
        self.validation_contra_losses = self.checkpoint.get('validation_contra_losses', [])
        return self.validation_contra_losses


def forward_model(model, dataset, query_array=None, batch_size=16,
                  use_matrix_index=True, is_weight_in_label=False):
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
                if use_matrix_index:
                    query_vectors = query_array_tensor[matrix_indices].to(images.device)
                    weights = model(images, query_vectors).squeeze()
                else:
                    query_vectors = query_array_tensor.repeat(batch_size, 1, 1).to(images.device)
                    # print(f'query_vector shape: {query_vectors.shape}')
                    # print(f'images shape: {images.shape}')
                    if query_vectors.size(0) != images.size(0):
                        query_vectors = query_vectors[:images.size(0), :, :]
                    weights = model(images, query_vectors)
            else:
                images, labels = data
                weights = model(images).squeeze()

            #images = images.to(next(model.parameters()).device)
            all_weights.extend(weights.cpu().tolist())
            all_labels.extend(labels.cpu().tolist() if torch.is_tensor(labels) else labels)

    # Normalize weights
    if is_weight_in_label:
        weights_tensor = torch.tensor(all_labels)
    else:
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
            #query_vectors = query_array_tensor[matrix_indices].to(images.device)
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



