import torch
from torch.utils.data import DataLoader
from .loss_function import CosineNegativePairLoss
import numpy as np
from operator import itemgetter


class Trainer:
    def __init__(self, model, criterion, optimizer, device, accumulation_steps=1,
                 query_array=None, is_contrastive_learning=False, is_selective_layers=False,
                 query_encoder=None, query_permutator=None, series_ids=None, is_feature_L1=False,
                 is_retinal_dataset=True,
                 margin=0.1, temperature=0.1, lambda_l1=0.01, contrastive_factor=0.01,
                 l1_weight=0.01):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.is_retinal_dataset = is_retinal_dataset

        self.is_contrastive_learning = is_contrastive_learning
        if self.is_contrastive_learning:
            if query_encoder is None or query_permutator is None or series_ids is None:
                raise ValueError("Not enough inputs in Trainer to perform contrastive learning.")

            self.query_encoder = query_encoder
            self.query_permutator = query_permutator
            self.series_ids = series_ids
            self.neg_contra_loss_fn = CosineNegativePairLoss(margin=margin, temperature=temperature)
        self.is_selective_layers = is_selective_layers
        self.is_feature_L1 = is_feature_L1
        self.lambda_l1 = lambda_l1
        self.contrastive_factor = contrastive_factor
        self.l1_weight = l1_weight

        if query_array is not None:
            self.is_query_array = True
            if self.is_selective_layers:
                self.query_array = torch.from_numpy(query_array)
            else:
                self.query_array = torch.from_numpy(query_array).unsqueeze(1)
        else:
            self.is_query_array = False

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
                elif self.is_selective_layers:
                    loss = self._process_batch_with_query_selective(data)
                else:
                    loss = self._process_batch_with_query(data)
            else:
                loss = self._process_batch(data)

            if loss is not None:
                self._update_parameters(loss)
                total_train_loss += loss.item()
            else:
                raise ValueError(f"Loss is None for batch {batch_idx}. Check your model's output and loss function.")

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                self.optimizer.step()  # Perform optimization step
                self.optimizer.zero_grad()  # Zero gradients for the next accumulation

        avg_train_loss = total_train_loss / len(train_loader)
        return avg_train_loss

    def _process_batch(self, data):
        if self.is_retinal_dataset:
            input_matrices, targets, _ = data
        else:
            input_matrices, targets = data
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        targets = targets.unsqueeze(1)
        outputs = self.model(input_matrices)
        loss = self._compute_loss(outputs, targets)
        return loss

    def _process_batch_with_query(self, data):
        input_matrices, targets, matrix_indices = data
        # print(f'matrix_indices shape: {matrix_indices.shape}')
        # print(f'query_array shape: {self.query_array.shape}')
        query_vectors = self.query_array[matrix_indices.squeeze(), :, :]
        # print(f'query_vector shape: {query_vectors.shape}')
        query_vectors = query_vectors.float().to(self.device)
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)
        outputs, _ = self.model(input_matrices, query_vectors)
        try:
            assert outputs.shape == targets.shape
        except Exception as e:
            print(e)
            print(f'outputs shape: {outputs.shape}')
            print(f'targets shape: {targets.shape}')

        loss = self._compute_loss(outputs, targets)
        if torch.isnan(loss).any():
            print(f"loss: {loss} \n")
            print(f"outputs: {outputs} \n")
            print(f"targets: {targets} \n")
            raise RuntimeError("Output value contain nan")

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
            contra_loss += self.contrastive_factor*torch.matmul(targets.T, self.neg_contra_loss_fn(perm_embedding.view(num_batch, -1),
                                                                           outputs_embedding.view(num_batch, -1)))

        return self._compute_loss(outputs_predict, targets) + contra_loss

    def _process_batch_with_query_selective(self, data):
        input_matrices, targets, matrix_indices = data
        query_vectors = self.query_array[matrix_indices]
        input_matrices, targets = input_matrices.to(self.device), targets.to(self.device)

        dataset_ids = query_vectors[:, 0].to(self.device)
        neuron_ids = query_vectors[:, 3].to(self.device)
        if self.is_feature_L1:
            outputs_predict, feature_gamma, spatial_gamma = self.model(input_matrices, dataset_ids, neuron_ids)
            l1_loss = self.l1_weight*(self._l1_regularization(feature_gamma, self.lambda_l1) +
                                      self._l1_regularization(spatial_gamma, self.lambda_l1))
        else:
            outputs_predict = self.model(input_matrices, dataset_ids, neuron_ids)
            raise ValueError(f"Temporal close {neuron_ids}.")
            l1_loss = _l1_regularization(self.model.spamap.spatial_embedding.weight[neuron_ids], self.lambda_l1) + \
                      _l1_regularization(self.model.feamap.channel_embedding.weight[neuron_ids], self.lambda_l1)

        return self._compute_loss(outputs_predict, targets) + l1_loss

    def _compute_loss(self, outputs, targets):
        return self.criterion(outputs.squeeze(), targets.squeeze())

    def _update_parameters(self, loss):
        loss.backward()  # Compute the backward pass only

    def _l1_regularization(self, weight, lambda_l1):
        return lambda_l1 * torch.abs(weight).sum()


class Evaluator(Trainer):
    def __init__(self, model, criterion, device,
                 query_array=None, is_contrastive_learning=False, is_selective_layers=False,
                 query_encoder=None, query_permutator=None, series_ids=None, is_feature_L1=False,
                 is_retinal_dataset=True,
                 margin=0.1, temperature=0.1, lambda_l1=0.01, contrastive_factor=0.01,
                 l1_weight=0.01):
        # Initialize the parent class without an optimizer as it's not needed for evaluation
        super().__init__(model, criterion, None, device, query_array=query_array,
                         is_contrastive_learning=is_contrastive_learning, is_selective_layers=is_selective_layers,
                         query_encoder=query_encoder, series_ids=series_ids, query_permutator=query_permutator,
                         is_retinal_dataset=is_retinal_dataset,
                         margin=margin, temperature=temperature, lambda_l1=lambda_l1, is_feature_L1=is_feature_L1,
                         contrastive_factor=contrastive_factor, l1_weight=l1_weight)

    def evaluate(self, eval_loader):
        self.model.eval()  # Set the model to evaluation mode
        total_eval_loss = 0
        with torch.no_grad():  # Ensure no gradients are computed
            for batch_idx, data in enumerate(eval_loader):
                if self.is_query_array:
                    if self.is_contrastive_learning:
                        loss = self._process_batch_with_query_contrast(data)
                    elif self.is_selective_layers:
                        loss = self._process_batch_with_query_selective(data)
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
                    validation_losses, validation_contra_losses=None,
                    file_path=None):
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
        # self.checkpoint = torch.load(checkpoint_path)

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

    def load_validation_contra_losses(self):
        """ Return the list of recorded validation losses. """
        self.validation_contra_losses = self.checkpoint.get('validation_contra_losses', [])
        return self.validation_contra_losses

# task: needed to add the demonstration of original dataset
# (1) just the label results
# (2) the trained model results
def forward_model(model, dataset, query_array=None, batch_size=16,
                  use_matrix_index=True, is_weight_in_label=False, logger=None,
                  model_type=None, is_retinal_dataset=False, is_rescale_image=False,
                  presented_cell_id=None):
    model.eval()  # Set the model to evaluation mode

    all_weights = []
    all_labels = []
    all_batch_idx = []
    all_within_batch_idx = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    is_adding = False

    # Prepare query array if provided
    query_array_tensor = torch.from_numpy(query_array).unsqueeze(1).float() if query_array is not None else None
    use_query = query_array_tensor is not None

    if model_type == 'FiLMCNN':
        query_array_tensor = torch.from_numpy(query_array)
        is_adding = True

    # First pass: Compute weights for all images
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if use_query:
                images, labels, matrix_indices = data
                if batch_size != images.size(0):
                    is_adding = False
                    continue
                if is_rescale_image:
                    images = images*2-1

                if use_matrix_index:
                    query_vectors = query_array_tensor[matrix_indices].to(images.device)
                    weights = model(images, query_vectors).squeeze()
                if model_type == 'FiLMCNN':
                    query_vectors = query_array_tensor.repeat(batch_size, 1)
                    #input_matrices = input_matrices.to(images.device)
                    dataset_ids = query_vectors[:, 0].to(images.device)
                    neuron_ids = query_vectors[:, 3].to(images.device)
                    matrix_indices = matrix_indices.to(images.device)
                    labels = labels.to(images.device)
                    assert neuron_ids.size(0) == images.size(0)
                    assert neuron_ids.size(0) == batch_size
                    assert neuron_ids.size(0) == dataset_ids.size(0)
                    weights, _, _ = model(images, dataset_ids, neuron_ids)
                    mask = (matrix_indices == neuron_ids)
                    within_idx = torch.arange(weights.size(0)).to(images.device)
                    weights = weights[mask]
                    labels = labels[mask]
                    within_idx = within_idx[mask]

                    # if logger is not None:
                    #    logger.info(f'Finish {batch_idx}/{len(dataloader)} \n')
                elif is_retinal_dataset:
                    query_vectors = query_array_tensor.repeat(batch_size, 1, 1).to(images.device)
                    weights, _ = model(images, query_vectors)
                    matrix_indices = matrix_indices.to(images.device)
                    labels = labels.to(images.device)
                    mask = (matrix_indices == presented_cell_id)
                    within_idx = torch.arange(weights.size(0)).to(images.device)
                    weights = weights[mask]
                    labels = labels[mask]
                    within_idx = within_idx[mask]
                    is_adding = True

                else:
                    query_vectors = query_array_tensor.repeat(batch_size, 1, 1).to(images.device)
                    # print(f'query_vector shape: {query_vectors.shape}')
                    # print(f'images shape: {images.shape}')
                    if query_vectors.size(0) != images.size(0):
                        query_vectors = query_vectors[:images.size(0), :, :]
                    weights, _ = model(images, query_vectors)
            else:
                if is_retinal_dataset:
                    images, labels, _ = data
                    weights = model(images).squeeze()
                else:
                    images, labels = data
                    weights, _ = model(images).squeeze()

            # images = images.to(next(model.parameters()).device)
            # print(f'weights type: {type(weights)}')
            # print(f'weights shape: {weights.shape}')
            weights_list = weights.cpu().tolist()

            if is_adding:
                within_idx_list = within_idx.cpu().tolist()
                batch_idx_list = [batch_idx] * len(weights_list)
                all_within_batch_idx.extend(within_idx_list)
                all_batch_idx.extend(batch_idx_list)
                batch_indices_tensor = torch.tensor(all_batch_idx)
                within_batch_indices_tensor = torch.tensor(all_within_batch_idx)

            all_weights.extend(weights_list)
            all_labels.extend(labels.cpu().tolist() if torch.is_tensor(labels) else labels)

    if logger:
        logger.info('finished weights model outputs')
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
    for batch_idx, data in enumerate(dataloader):
        if use_query:
            images, _, matrix_indices = data
            if batch_size != images.size(0):
                continue
            if model_type == 'FiLMCNN':
                mask = batch_indices_tensor == batch_idx
                weights_batch = normalized_weights[mask].to(images.device).view(-1, 1, 1, 1, 1)
                images = images[within_batch_indices_tensor[mask]]
            elif is_retinal_dataset:
                mask = batch_indices_tensor == batch_idx
                weights_batch = normalized_weights[mask].to(images.device).view(-1, 1, 1, 1, 1)
                images = images[within_batch_indices_tensor[mask]]
            else:
                weights_batch = normalized_weights[idx:idx + images.size(0)].to(images.device).view(-1, 1, 1, 1, 1)

            # print(f"Batch ID: {batch_idx}/{len(dataloader)}\n")
            # print(f'weight_batch shape: {weights_batch.shape} \n')
            # print(f'images shape: {images.shape} \n')
            weighted_images = images * weights_batch

        else:
            if is_retinal_dataset:
                images, _, _ = data
                #print(f'images size:{images.shape}')
            else:
                images, _ = data

            if batch_size != images.size(0):
                continue
            images = images.to(next(model.parameters()).device)
            weights_batch = normalized_weights[idx:idx + images.size(0)].to(images.device).view(-1, 1, 1, 1, 1)
            #print(f'weights_batch size:{weights_batch.shape}')
            weighted_images = images * weights_batch

        #print(f'weighted images size:{weighted_images.shape}')
        batch_sum = weighted_images.sum(dim=0)  # Sum over the batch
        #print(f'batch sum size:{batch_sum.shape}')

        if weighted_sum is None:
            print(f'weighted sum size:{weighted_sum.shape}')
            weighted_sum = batch_sum
        else:
            weighted_sum += batch_sum

        idx += images.size(0)

    return weighted_sum, all_weights, all_labels
