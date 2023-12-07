import numpy as np
import matplotlib.pyplot as plt
import parameters.params_20231206_01 as p

def generate_2d_gaussian(size, mean, cov):
    """
    Generate a 2D Gaussian distribution matrix.

    Args:
    size (tuple): Size of the matrix (n, n).
    mean (tuple): Mean of the Gaussian (x_mean, y_mean).
    cov (tuple): Covariance of the Gaussian (x_cov, y_cov).

    Returns:
    numpy.ndarray: 2D Gaussian distribution matrix.
    """
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    x, y = np.meshgrid(x, y)

    d = np.dstack([x, y])
    gaussian_matrix = np.exp(-0.5 * np.sum(np.dot(d - mean, np.linalg.inv(cov)) * (d - mean), axis=2))

    return gaussian_matrix

result_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
cfid = 2
# Parameters for the Gaussian
mean = (0.1, -0.2)  # Mean centered at (0, 0)
cov = np.array([[0.12, 0.05], [0.04, 0.03]])  # Covariance
# Generate the target matrix
target_matrix = np.array([generate_2d_gaussian((input_width, input_height), mean, cov) * time_point for time_point in freqf_t[:input_depth]])
#print(target_matrix.shape)
# Plot the matrix
num_cols = 5  # Number of columns in the subplot grid
num_rows = p.input_depth // num_cols + int(p.input_depth % num_cols > 0)

plt.figure(figsize=(15, num_rows * 3))

for i in range(p.input_depth):
    ax = plt.subplot(num_rows, num_cols, i + 1)
    image = ax.imshow(target_matrix[i], cmap='viridis', vmin=-0.4, vmax=0.5)
    plt.title(f'Time Frame {i}')
    plt.axis('off')

    # Add a colorbar for each subplot
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
plt.savefig(f'{result_dir}check_figure{cfid}.png')