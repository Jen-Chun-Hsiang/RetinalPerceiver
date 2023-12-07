from utils.utils import create_checkpoint_filename

# Example usage
checkpoint_dir = './checkpoints'  # Directory where checkpoints are saved
new_checkpoint_file = create_checkpoint_filename(checkpoint_dir)
print(new_checkpoint_file)