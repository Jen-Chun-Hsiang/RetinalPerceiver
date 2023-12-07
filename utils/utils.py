import torch
import os
from datetime import datetime

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

def create_checkpoint_filename(checkpoint_dir, file_prefix='model_checkpoint'):
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d')

    # Check for existing files and determine the next series number
    existing_files = os.listdir(checkpoint_dir)
    max_series_num = 0
    for file in existing_files:
        if file.startswith(file_prefix) and timestamp in file:
            parts = file.split('_')
            try:
                series_num = int(parts[-1].split('.')[0])  # Extract series number
                max_series_num = max(max_series_num, series_num)
            except ValueError:
                # Handle the case where the filename format is unexpected
                continue

    next_series_num = max_series_num + 1

    # Create new checkpoint filename
    new_filename = f"{file_prefix}_{timestamp}_{next_series_num}.pth"
    return new_filename

