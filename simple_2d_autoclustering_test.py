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
import matplotlib.patches as patches
from datetime import datetime
import argparse
import logging
from utils.simple_2d import GaussianDataset, CrossAttentionNet, compute_sta, CrossAttentionNetAlt

import pandas as pd
import scipy.io


def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')

    # dataset
    parser.add_argument('--num_A', type=int, default=20, help='Number of data points - group A')
    parser.add_argument('--num_B', type=int, default=4, help='Number of data points - group B')
    parser.add_argument('--rng_seed', type=int, default=45, help='assign a random seed')
    parser.add_argument('--is_unknown_center_new', action='store_true', help='provide unique type class for new centers')
    parser.add_argument('--image_size', type=int, default=32, help='Input image size')
    parser.add_argument('--num_total_types', type=int, default=5, help='Number of total types')
    parser.add_argument('--num_known_types', type=int, default=3, help='Number of known types')
    parser.add_argument('--boundary', type=int, default=4, help='image boundary to generate a receptive field')
    parser.add_argument('--num_center_pos', type=int, default=5, help='Number of different center position in training')
    parser.add_argument('--num_worker', type=int, default=0, help='Use to offline loading data in batch')
    # Model
    parser.add_argument('--early_tau', type=float, default=1e-7, help='Temperature for gumbel tau in early training stage')
    parser.add_argument('--late_tau', type=float, default=1.0, help='Temperature for gumbel tau in late training stage')
    parser.add_argument('--type_embed_dim', type=int, default=5, help='Number of dimension of the embedding of the types')
    parser.add_argument('--tau_switch_epoch', type=int, default=250, help='Epoch number to switch gumbel tau from early to late')
    parser.add_argument('--cell_type_encoding_dim', type=int, default=3, help='Number of low dimension cell type embedding')
    parser.add_argument('--is_alt_model', action='store_true', help='Enable consistency loss')
    # Training
    parser.add_argument('--is_GPU', action='store_true', help='Using GPUs for accelaration')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of total epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='Number of epochs to save a checkpoints')
    parser.add_argument('--cluster_weight', type=float, default=0.0001, help='Weights of the cluster loss')
    parser.add_argument('--num_samples', type=int, default=20000, help='Number of data samples in the dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--is_showing_STA', action='store_true', help='generate sta for each cell')
    parser.add_argument('--consistency_weight', type=float, default=1e-1, help='panelty strength of the consistency losses')
    return parser.parse_args()

def main():
    args = parse_args()
    filename_fixed = args.experiment_name
    specific_known = [
        {"center": [16, 16], "theta": 1.0,               "eig1": 10, "eig2": 2, "type_id": 0, "surround_strength": 0.1},
        {"center": [16, 16], "theta": 1.0+math.pi/2,     "eig1": 10, "eig2": 2, "type_id": 1, "surround_strength": 0.1},
        {"center": [16, 16], "theta": 1.0 + math.pi / 2, "eig1": 2,  "eig2": 2, "type_id": 2, "surround_strength": 0.8},
        {"center": [16, 16], "theta": 1.0 + math.pi / 2, "eig1": 6,  "eig2": 6, "type_id": 3, "surround_strength": 0.8},
        {"center": [16, 16], "theta": 1.0 + math.pi / 4, "eig1": 10, "eig2": 2, "type_id": 4, "surround_strength": 0.2},
        # add more as needed...
    ]
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
    is_applied_low_dim_type_encoding = False

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
                              is_unknown_center_new=is_unknown_center_new, specific_known_cells=specific_known,
                              num_total_types=num_total_types, num_known_types=num_known_types,
                              boundary=boundary, num_center_pos=args.num_center_pos)
    for i in range(5):
        dataset.plot_sample(i, save_folder=savefig_dir, save_name=f'{filename_fixed}_plot_cell_RF.png')
    dataset.print_cell_table()

    if args.num_worker == 0:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_worker, pin_memory=True, persistent_workers=False)

    init_type_num = args.num_A + args.num_B
    if args.is_alt_model:
        model = CrossAttentionNetAlt(d_model=32, hidden_dim=32, center_B=10, num_total_types=num_total_types,
                              type_embed_dim=args.type_embed_dim, init_type_num=init_type_num,
                              cell_type_encoding_dim=args.cell_type_encoding_dim)
    else:
        model = CrossAttentionNet(d_model=32, hidden_dim=32, center_B=10, num_total_types=num_total_types,
                                  type_embed_dim=args.type_embed_dim, init_type_num=init_type_num,
                                  cell_type_encoding_dim=args.cell_type_encoding_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    mse_loss = nn.MSELoss()

    losses_dict = {"epochs": [], "total_loss": [], "reg_loss": [], "cluster_loss": [], "consistency_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_total_loss = 0.0
        running_reg_loss = 0.0
        running_cluster_loss = 0.0
        running_consistency_loss = 0.0
        total_samples = 0

        for batch in loader:
            optimizer.zero_grad()

            # Unpack the batch; note that your dataset should now provide the keys below.
            images = batch['image'].to(device)                 # [B, 1, H, W]
            target = batch['target'].to(device)                  # [B]
            query_center = batch['query_center'].to(device)        # [B, 2]
            is_center_known = batch['is_center_known'].to(device)    # [B] bool
            unknown_center_id = batch['unknown_center_id'].to(device) # [B] long
            type_gt = batch['type_gt'].to(device)                    # [B] long
            is_type_known = batch['is_type_known'].to(device)        # [B] bool
            cell_idx = batch['cell_idx'].to(device)                  # [B] long

            if (epoch + 1) > args.tau_switch_epoch:
                tau = args.late_tau
            else:
                tau = args.early_tau

            cluster_weight = args.cluster_weight
            alpha = (epoch/num_epochs) ** 2
            beta = alpha*args.consistency_weight


            if args.is_alt_model:
                target_pred, global_entropy, consistency_loss = \
                    model(images, query_center, is_center_known, unknown_center_id, type_gt, is_type_known, cell_idx,
                          tau=tau, alpha=alpha)
            else:
                target_pred, global_entropy = model(images, query_center, is_center_known, unknown_center_id, type_gt,
                                                    is_type_known, cell_idx, tau=tau, alpha=alpha)
                consistency_loss = torch.tensor(0.0, device=images.device)

            loss_reg = mse_loss(target_pred, target)
            loss_cluster = cluster_weight * global_entropy

            total_loss = loss_reg + loss_cluster + beta * consistency_loss

            total_loss.backward()
            optimizer.step()
            # Clamp the unknown center embedding weights as in the original code
            model.unknown_embedding.weight.data.clamp_(-0.999, 0.999)

            batch_size = images.size(0)
            running_total_loss += total_loss.item() * batch_size
            running_reg_loss += loss_reg.item() * batch_size
            running_cluster_loss += loss_cluster.item() * batch_size
            running_consistency_loss += consistency_loss.item() * batch_size
            total_samples += batch_size

        # Step the scheduler after each epoch
        scheduler.step()

        # Compute average losses per sample for the epoch
        epoch_total_loss = running_total_loss / total_samples if total_samples > 0 else 0.0
        epoch_reg_loss = running_reg_loss / total_samples if total_samples > 0 else 0.0
        epoch_cluster_loss = running_cluster_loss / total_samples if total_samples > 0 else 0.0
        epoch_consistency_loss = running_consistency_loss / total_samples if total_samples > 0 else 0.0

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {epoch_total_loss:.6f}, "
                     f"Reg Loss: {epoch_reg_loss:.6f}, \n"
                     f"Consistency loss: {epoch_consistency_loss: 6f}, Cluster Loss: {epoch_cluster_loss:.6f} \n")
        # Store losses for this epoch.
        losses_dict["epochs"].append(epoch + 1)
        losses_dict["total_loss"].append(epoch_total_loss)
        losses_dict["reg_loss"].append(epoch_reg_loss)
        losses_dict["cluster_loss"].append(epoch_cluster_loss)
        losses_dict["consistency_loss"].append(epoch_consistency_loss)

        # Save a checkpoint every 'checkpoint_interval' epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(savemodel_dir, f"{filename_fixed}_checkpoint_epoch_{epoch+1}.pth")
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
    reg_loss = np.array(losses_dict["reg_loss"])
    cluster_loss = np.array(losses_dict["cluster_loss"])

    # Create a single plot for all loss types.
    plt.figure(figsize=(8, 6))

    # Plot total loss.
    plt.plot(epochs, total_loss, marker='o', linestyle='-', label="Total Loss")

    # Plot regression (main task) loss.
    plt.plot(epochs, reg_loss, marker='s', linestyle='--', color='g', label="Regression Loss")

    # Plot clustering (global entropy regularization) loss.
    plt.plot(epochs, cluster_loss, marker='d', linestyle='-.', color='r', label="Cluster Loss")

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

    if args.is_showing_STA:
        for i in range(num_A + num_B):
            sta_image, all_outputs = compute_sta(model, dataset, i, num_stimuli=10000, threshold=None, device=device)

            # Plot the resulting STA image.
            plt.figure(figsize=(5, 5))
            plt.imshow(sta_image, cmap='viridis')
            plt.title(f"STA for Cell {i}")
            plt.colorbar()

            save_name = f'{filename_fixed}_trained_STA_{i}.png'
            save_name = os.path.join(savefig_dir, f"{save_name}")
            plt.savefig(save_name, dpi=300, bbox_inches="tight")

    if hasattr(dataset, "type_known_flags"):
        type_known_flags = dataset.type_known_flags
    else:
        type_known_flags = [cell["type_id"] < dataset.num_known_types for cell in dataset.cell_properties]

    # Ensure we know on which device the model is located.
    device = next(model.parameters()).device

    # Create a list of dictionaries to store cell information
    cells_info = []
    for i, cell in enumerate(dataset.cell_properties):
        target_type = cell["type_id"]
        # Determine whether this cell's type was provided (i.e. "known")
        type_known = type_known_flags[i]

        center = cell["center"]
        center_norm = (center / dataset.image_size) * 2 - 1  # normalized
        is_center_known = i < dataset.A
        if is_center_known is False:
            unknown_center_id = i - dataset.A
            learned_center_embedding = model.unknown_embedding(
                torch.tensor(unknown_center_id, dtype=torch.long).to(device)
            ).detach().cpu().numpy()
        else:
            learned_center_embedding = center_norm

        if type_known:
            # For known cells, the query uses the ground-truth type,
            # so the "learned" type is simply the corresponding fixed embedding.
            learned_embedding = model.type_embedding.weight[target_type].detach().cpu().numpy()
            predicted_type = target_type
        else:
            # For unknown cells, we use the learned cell_type_logits.
            # Assume that unknown cells are those with index >= dataset.A,
            # and the lookup index for cell_type_logits is computed as (cell_id - dataset.A).
            unknown_index = i   # - dataset.A
            if is_applied_low_dim_type_encoding:
                # Use the learnable low-dimensional encoding path.
                encoding = model.cell_type_encoding(
                    torch.tensor(unknown_index, dtype=torch.long).to(device)
                )
                logits = model.cell_type_logits_proj(encoding)
            else:
                # Use the original cell_type_logits.
                logits = model.cell_type_logits(
                    torch.tensor(unknown_index, dtype=torch.long).to(device)
                )
            learned_embedding = logits.detach().cpu().numpy()\
            # Use Gumbel-Softmax (with tau=1.0, hard=True) to get a nearly one-hot vector.
            one_hot = F.gumbel_softmax(logits, tau=0.001, hard=True)
            predicted_type = int(torch.argmax(logits).item())

        cells_info.append({
            "Cell ID": i,
            "Target Type": target_type,
            "Type Known": type_known,
            "Predicted Type": predicted_type,
            "Learned Type Embedding": np.array2string(learned_embedding, precision=3),
            "is_center_known": is_center_known,
            "Targeted Center Embedding": np.array2string(center_norm),
            "Learned Center Embedding": np.array2string(learned_center_embedding),
        })

    # Create a DataFrame to display the results
    df = pd.DataFrame(cells_info)
    selected_columns = ["Cell ID", "Target Type", "Type Known", "Predicted Type"]
    logging.info(f'df: {df[selected_columns]} \n')

    save_name = os.path.join(savefig_dir, f'{filename_fixed}_learned_logit.png')
    plot_cell_type_logits_heatmap(dataset, model, device, save_name=save_name,
                                  is_applied_low_dim_type_encoding=is_applied_low_dim_type_encoding)


def plot_cell_type_logits_heatmap(dataset, model, device, save_name=None, is_applied_low_dim_type_encoding=True):
    """
    Plots a heatmap for cell_type_logits for unknown cells.
    For each unknown cell:
      - A white rectangle highlights the predicted cell type (i.e. the argmax of the logits).
      - If the predicted type does not match the target type, a red rectangle is drawn around the target type.

    Args:
        dataset: Dataset object containing:
            - cell_properties: list of dicts with cell info (each should have "type_id").
            - image_size: scalar used for normalization.
            - A: number of known cells (unknown cells follow).
        model: Model object with a callable attribute 'cell_type_logits' that returns logits.
        device: Torch device to use.
        save_name: Filename to save the plot, if provided.
    """
    unknown_indices = [i for i, flag in enumerate(dataset.type_known_flags) if not flag]

    logits_matrix = []
    target_types = []
    predicted_types = []

    # For each unknown cell, compute the logits from cell_type_logits.
    # Here we assume that the lookup index for the logits is computed as (i - dataset.A)
    for j, i in enumerate(unknown_indices):
        if is_applied_low_dim_type_encoding:
            encoding = model.cell_type_encoding(torch.tensor(i, dtype=torch.long).to(device))
            logits = model.cell_type_logits_proj(encoding)
        else:
            logits = model.cell_type_logits(torch.tensor(i, dtype=torch.long).to(device))

        logits = logits.detach().cpu().numpy()  # shape: [num_total_types]
        logits_matrix.append(logits)

        # Predicted type is the argmax of the logits.
        pred_type = int(np.argmax(logits))
        predicted_types.append(pred_type)

        # Retrieve the target type from cell_properties.
        target_types.append(dataset.cell_properties[i]["type_id"])

    # Convert lists to numpy arrays for sorting
    logits_matrix = np.array(logits_matrix)  # shape: (num_unknown_cells, num_total_types)
    target_types = np.array(target_types)
    predicted_types = np.array(predicted_types)

    # Sort the arrays by target_types
    sort_indices = np.argsort(target_types)
    logits_matrix = logits_matrix[sort_indices]
    target_types = target_types[sort_indices]
    predicted_types = predicted_types[sort_indices]

    # Create the heatmap
    plt.figure(figsize=(10, logits_matrix.shape[0] * 0.5 + 3))
    im = plt.imshow(logits_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(im, label="Logit Value")
    plt.xlabel("Cell Type Index")
    plt.ylabel("Unknown Cell Index (sorted by target type)")
    plt.title("Heatmap of cell_type_logits for Unknown Cells")

    ax = plt.gca()
    num_rows, num_cols = logits_matrix.shape

    # For each row (unknown cell), add rectangle annotations
    for row in range(num_rows):
        pred = predicted_types[row]
        target = target_types[row]

        # Draw a white rectangle around the predicted type
        rect_pred = patches.Rectangle((pred - 0.5, row - 0.5), 1, 1, linewidth=2,
                                      edgecolor='white', facecolor='none')
        ax.add_patch(rect_pred)

        # If the predicted type differs from the target, highlight the target type with a red rectangle
        if pred != target:
            rect_target = patches.Rectangle((target - 0.5, row - 0.5), 1, 1, linewidth=2,
                                            edgecolor='red', facecolor='none')
            ax.add_patch(rect_target)

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
    else:
        plt.show()



if __name__ == '__main__':
    main()





