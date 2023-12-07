# Parameters
input_depth = 20  # Size of the matrix
input_height = 30  # Size of the matrix
input_width = 40  # Size of the matrix
input_channels = 1  # Number of channels in the input
total_length = 10000  # Total number of samples
train_length = int(0.8 * total_length)  # 80% for training
val_length = total_length - train_length  # 20% for validation
hidden_size = 128
output_size = 1  # Output is a single value
learning_rate = 0.001
weight_decay = 0.01    # Set an appropriate weight decay if needed
batch_size = 64
epochs = 30
num_head = 4
num_iter = 1
num_latent = 16
num_band = 10