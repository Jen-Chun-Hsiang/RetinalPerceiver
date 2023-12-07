import torch


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
