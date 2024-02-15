import numpy as np
from PIL import Image
import torch
from skimage.transform import resize
import matplotlib.pyplot as plt


def load_png_to_numpy(png_path):
    """Load PNG image and convert to a NumPy array."""
    image = Image.open(png_path)
    return np.array(image)


def load_pth_to_numpy(pth_path):
    """Load PyTorch tensor and convert to a NumPy array."""
    tensor = torch.load(pth_path)
    # Ensure it's a CPU tensor
    tensor = tensor.cpu()
    return tensor.numpy()


def compare_images(image1_np, image2_np):
    """Compare two images by computing the correlation coefficient between their pixels."""
    print(f'image1 shape {image1_np.shape}')
    print(f'image2 shape {image2_np.shape}')
#    if image1_np.shape != image2_np.shape:
#        raise ValueError("Images must have the same shape to compare.")
    image1_flat = image1_np.flatten()
    image2_flat = image2_np.flatten()
    corr_matrix = np.corrcoef(image1_flat, image2_flat)
    return corr_matrix[0, 1]


def plot_images(image1_np, image2_np, correlation_coefficient, save_path):
    """Plot two images side by side, their difference, and include the correlation coefficient in the title."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image1_np)
    axs[0].set_title('Image 1')
    axs[0].axis('off')

    axs[1].imshow(image2_np)
    axs[1].set_title('Image 2')
    axs[1].axis('off')

    diff_image = np.abs(image1_np - image2_np)
    axs[2].imshow(diff_image)
    axs[2].set_title('Absolute Difference')
    axs[2].axis('off')

    plt.suptitle(f"Correlation Coefficient: {correlation_coefficient:.4f}", fontsize=16)
    plt.savefig(save_path)
    plt.show()


# Example usage
png_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/experiment_1/session_3/frame_5001.png'
pth_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/TrainingSet/Stimulus/experiment_1/session_3/frame_5001.pt'
save_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/check_image_5001.png'  # Specify your save path here

# Load and resize images
image_png_np = load_png_to_numpy(png_path).squeeze()
image_pth_np = load_pth_to_numpy(pth_path).squeeze()

# Compare images and plot them
correlation_coefficient = compare_images(image_png_np, image_pth_np)
plot_images(image_png_np, image_pth_np, correlation_coefficient, save_path)

print(f"Figure saved to {save_path}")
