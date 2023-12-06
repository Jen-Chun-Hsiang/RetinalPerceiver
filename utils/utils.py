import torch


# Function to evaluate the model on validation data
def validate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to track gradients during validation
        for input_matrices, targets in dataloader:
            targets = targets.view(targets.size(0), -1)
            outputs = model(input_matrices)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)