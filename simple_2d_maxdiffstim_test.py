import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import logging
from utils.simple_2d import GaussianDataset, compute_sta, CrossAttentionNet_POS
from utils.simple_2d_helper import adaptive_grad_clip, SharedPerturbationOptimizer

import pandas as pd
import scipy.io


def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')
    parser.add_argument('--num_A', type=int, default=32, help='Number of data points - group A')
    parser.add_argument('--num_B', type=int, default=4, help='Number of data points - group B')
    parser.add_argument('--rng_seed', type=int, default=48, help='assign a random seed')
    parser.add_argument('--is_unknown_center_new', action='store_true', help='provide unique type class for new centers')
    parser.add_argument('--image_size', type=int, default=32, help='Input image size')
    parser.add_argument('--num_total_types', type=int, default=7, help='Number of total types')
    parser.add_argument('--num_known_types', type=int, default=3, help='Number of known types')
    parser.add_argument('--boundary', type=int, default=4, help='image boundary to generate a receptive field')
    # Model
    parser.add_argument('--directional_loss_weight', type=float, default=0.0, help='R2 - R1 difference')
    # Training
    parser.add_argument('--is_GPU', action='store_true', help='Using GPUs for accelaration')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of total epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='Number of epochs to save a checkpoints')
    parser.add_argument('--num_samples', type=int, default=20000, help='Number of data samples in the dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--is_AGC', action='store_true', help='Add gradient clipping to make convergent to x, y more efficient')
    parser.add_argument('--is_noisy_grad', action='store_true', help='Add noisy gradient to prevent vanish gradient landscape')
    parser.add_argument('--noise_grad_std', type=float, default=1e-4, help='Std of gradient noise')
    return parser.parse_args()


def main():
    args = parse_args()
    filename_fixed = args.experiment_name
    specific_known_1 = [
        {"center": [16, 16], "theta": 1.0,               "eig1": 10, "eig2": 2, "type_id": 0, "surround_strength": 0.2},
        {"center": [16, 16], "theta": 1.0+math.pi/2,     "eig1": 10, "eig2": 2, "type_id": 1, "surround_strength": 0.2},
    ]
    specific_known_2 = [
        {"center": [16, 16], "theta": 1.0 + math.pi / 2, "eig1": 2, "eig2": 2, "type_id": 2, "surround_strength": 0.8},
        {"center": [16, 16], "theta": 1.0 + math.pi / 2, "eig1": 6, "eig2": 6, "type_id": 3, "surround_strength": 0.8},
    ]
    specific_known = specific_known_1
    output_mode = 'A'
    num_A = args.num_A
    num_B = args.num_B
    seed = args.rng_seed
    is_unknown_center_new = args.is_unknown_center_new
    image_size = args.image_size
    num_total_types = args.num_total_types
    num_known_types = args.num_known_types
    boundary = args.boundary
    num_epochs = args.num_epochs
    checkpoint_interval = args.checkpoint_interval

    # Folders
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'

    os.makedirs(saveprint_dir, exist_ok=True)  # Ensure folder exists
    os.makedirs(savefig_dir, exist_ok=True)  # Ensure folder exists
    os.makedirs(savemodel_dir, exist_ok=True)  # Ensure folder exists
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Construct the full path for the log file
    log_filename = os.path.join(saveprint_dir, f'{filename_fixed}_training_log_{timestr}.txt')

    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f'start logging... \n')

    if args.is_GPU:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        logging.info(f'set up GPU operation \n')
    else:
        device = 'cpu'
        logging.info(f'set up CPU operation \n')

    # randomization initiate
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = GaussianDataset(A=num_A, B=num_B, num_samples=args.num_samples, image_size=image_size,
                              is_unknown_center_new=is_unknown_center_new,
                              specific_known_cells=specific_known, num_total_types=num_total_types,
                              num_known_types=num_known_types,
                              boundary=boundary, output_mode=output_mode)
    for i in range(5):
        dataset.plot_sample(i, save_folder=savefig_dir, save_name=f'{filename_fixed}_plot_cell_RF.png')
    dataset.print_cell_table()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CrossAttentionNet_POS(d_model=32, hidden_dim=32, B=num_B)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    mse_loss = nn.MSELoss()

    losses_dict = {"epochs": [], "total_loss": [], "known_loss": [], "unknown_loss": []}

    for epoch in range(num_epochs):
        running_loss_total = 0.0
        running_loss_known = 0.0
        running_loss_unknown = 0.0
        total_samples = 0
        known_samples = 0
        unknown_samples = 0

        for batch in loader:
            optimizer.zero_grad()

            # Unpack the batch; note that your dataset should now provide the keys below.
            images = batch['image'].to(device)  # [B, 1, H, W]
            query = batch['query'].to(device)  # [B, 3]
            is_known = batch['is_center_known'].to(device)  # [B] bool
            unknown_id = batch['unknown_center_id'].to(device)
            target = batch['target'].to(device)  # [B]

            target_pred = model(images, query, is_known, unknown_id)

            loss = mse_loss(target_pred, target)

            loss.backward()

            if args.is_noisy_grad:
                epoch_progress = (num_epochs-epoch+1)/epoch
                if model.unknown_embedding.weight.grad is not None:
                    model.unknown_embedding.weight.grad.add_(
                        torch.randn_like(model.unknown_embedding.weight.grad) * args.noise_grad_std * epoch_progress
                    )

            if args.is_AGC:
                adaptive_grad_clip(model.unknown_embedding.parameters(), clip_factor=0.01)
            optimizer.step()

            model.unknown_embedding.weight.data.clamp_(-0.999, 0.999)

            batch_size = batch['image'].size(0)
            running_loss_total += loss.item() * batch_size
            total_samples += batch_size

            # Separate loss accumulation based on is_known
            mask_known = batch['is_center_known']
            mask_unknown = ~batch['is_center_known']

            if mask_known.any():
                images = batch['image'][mask_known].to(device)  # [B, 1, H, W]
                query = batch['query'][mask_known].to(device)  # [B, 3]
                is_known = batch['is_center_known'][mask_known].to(device)  # [B] bool
                unknown_id = batch['unknown_center_id'][mask_known].to(device)
                target = batch['target'][mask_known].to(device)  # [B]

                target_pred = model(images, query, is_known, unknown_id)
                loss_known = mse_loss(target_pred, target)
                running_loss_known += loss_known.item() * mask_known.sum().item()
                known_samples += mask_known.sum().item()

            if mask_unknown.any():
                images = batch['image'][mask_unknown].to(device)  # [B, 1, H, W]
                query = batch['query'][mask_unknown].to(device)  # [B, 3]
                is_known = batch['is_center_known'][mask_unknown].to(device)  # [B] bool
                unknown_id = batch['unknown_center_id'][mask_unknown].to(device)
                target = batch['target'][mask_unknown].to(device)  # [B]

                target_pred = model(images, query, is_known, unknown_id)
                loss_unknown = mse_loss(target_pred, target)
                running_loss_unknown += loss_unknown.item() * mask_unknown.sum().item()
                unknown_samples += mask_unknown.sum().item()

        scheduler.step()

        epoch_loss_total = running_loss_total / total_samples if total_samples > 0 else 0.0
        epoch_loss_known = running_loss_known / known_samples if known_samples > 0 else 0.0
        epoch_loss_unknown = running_loss_unknown / unknown_samples if unknown_samples > 0 else 0.0

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {epoch_loss_total:.6f}, "
                     f"KnownLoss: {epoch_loss_known:.6f}, Unknown Loss: {epoch_loss_unknown:.6f} \n")
        # Store loss values
        losses_dict["epochs"].append(epoch + 1)
        losses_dict["total_loss"].append(epoch_loss_total)
        losses_dict["known_loss"].append(epoch_loss_known)
        losses_dict["unknown_loss"].append(epoch_loss_unknown)

        # Save a checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(savemodel_dir, f"{filename_fixed}_checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses_dict
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}\n")

    # Retrieve the losses stored during training.
    epochs = np.array(losses_dict["epochs"])
    total_loss = np.array(losses_dict["total_loss"])
    known_loss = np.array(losses_dict["known_loss"])
    unknown_loss = np.array(losses_dict["unknown_loss"])

    # Create a single plot for all loss types.
    plt.figure(figsize=(8, 6))

    # Plot total loss.
    plt.plot(epochs, total_loss, marker='o', linestyle='-', label="Total Loss")

    # Plot regression (main task) loss.
    plt.plot(epochs, known_loss, marker='s', linestyle='--', color='g', label="Known location loss")

    # Plot clustering (global entropy regularization) loss.
    plt.plot(epochs, unknown_loss, marker='d', linestyle='-.', color='r', label="Unknown location loss")

    # Add labels and title.
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Show the plot.
    save_name = f'{filename_fixed}_losses.png'
    save_name = os.path.join(savefig_dir, f"{save_name}")
    plt.savefig(save_name, dpi=300, bbox_inches="tight")

    # 1. Extract the true (normalized) centers for the unknown cells directly.
    unknown_true_centers = []
    for cell in dataset.cell_properties[dataset.A:]:  # unknown cells are stored after the first A known cells.
        center = np.array(cell["center"])
        # Normalize the center as done in __getitem__: (center / image_size) * 2 - 1
        normalized_center = (center / dataset.image_size) * 2 - 1
        unknown_true_centers.append(normalized_center)
    unknown_true_centers = torch.tensor(unknown_true_centers, dtype=torch.float32)

    # 2. Get the learned queries directly from the unknown_embedding weight.
    # Ensure the model is on device and then move the weights to CPU for comparison.
    learned_queries = model.unknown_embedding.weight.data.to('cpu')

    # 3. Print comparison of each unknown cell.
    logging.info("Comparison of Unknown Queries (Learned) vs. True Centers: \n")
    for i, (true_center, learned) in enumerate(zip(unknown_true_centers, learned_queries)):
        logging.info(f"Unknown Cell {i}: True Center: {true_center.numpy()}, Learned Query: {learned.numpy()}")

    max_iter = 250
    image = torch.randn(1, image_size, image_size).unsqueeze(0).to(device)
    cell_idx = 0
    center_norm = (np.array(specific_known[cell_idx]['center']) / image_size) * 2 - 1  # normalized center
    type_id = dataset.cell_properties[cell_idx]["type_id"]
    query1 = torch.tensor(np.append(center_norm, type_id), dtype=torch.float32).unsqueeze(0).to(device)

    cell_idx = 1
    center_norm = (np.array(specific_known[cell_idx]['center']) / image_size) * 2 - 1  # normalized center
    type_id = dataset.cell_properties[cell_idx]["type_id"]
    query2 = torch.tensor(np.append(center_norm, type_id), dtype=torch.float32).unsqueeze(0).to(device)

    batch_size = 1
    # model(batch['image'], batch['query'], batch['is_known'], batch['unknown_id'])

    target1 = torch.tensor([1.0]).expand(batch_size, 1).to(device)  # Desired output for query1
    target2 = torch.tensor([-1.0]).expand(batch_size, 1).to(device)  # Desired output for query2

    optimizer = SharedPerturbationOptimizer(model, image, query1, query2, target1, target2, max_iter=max_iter,
                                            directional_loss_weight=args.directional_loss_weight)
    optimized_image_1 = optimizer.optimize()
    optimizer.print_model_outputs()
    optimizer.print_final_losses()

    logging.info('=====================================')

    target1 = torch.tensor([-1.0]).expand(batch_size, 1).to(device)  # Desired output for query1
    target2 = torch.tensor([1.0]).expand(batch_size, 1).to(device)  # Desired output for query2

    optimizer = SharedPerturbationOptimizer(model, image, query1, query2, target1, target2, max_iter=max_iter,
                                            directional_loss_weight=args.directional_loss_weight)
    optimized_image_2 = optimizer.optimize()
    optimizer.print_model_outputs()
    optimizer.print_final_losses()

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 1 row, 2 columns

    # First image
    axes[0, 0].imshow(image.cpu().squeeze(), cmap='gray', interpolation='nearest')
    axes[0, 0].set_title("Image 1")
    axes[0, 0].axis('off')  # Hide axes for better visualization

    # Second image
    axes[0, 1].imshow(optimized_image_1.cpu().squeeze(), cmap='gray', interpolation='nearest')
    axes[0, 1].set_title("Optimized Image")
    axes[0, 1].axis('off')

    # Second image
    axes[1, 0].imshow(optimized_image_2.cpu().squeeze(), cmap='gray', interpolation='nearest')
    axes[1, 0].set_title("Reversed optimized Image")
    axes[1, 0].axis('off')

    axes[1, 1].imshow((optimized_image_1 - optimized_image_2).cpu().squeeze(), cmap='gray', interpolation='nearest')
    axes[1, 1].set_title("Contrast optimized Image")
    axes[1, 1].axis('off')

    # Show the plot.
    save_name = f'{filename_fixed}_max_differential_stimuli.png'
    save_name = os.path.join(savefig_dir, f"{save_name}")
    plt.savefig(save_name, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
