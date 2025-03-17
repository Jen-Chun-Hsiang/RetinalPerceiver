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
from utils.simple_2d import GaussianDataset, CrossAttentionNet

import pandas as pd
import scipy.io


def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')

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
    ]  # MaxDiff03102502
    seed = 47
    is_unknown_center_new = False
    image_size = 32
    num_total_types = 5
    num_known_types = 3
    boundary = 4
    num_epochs = 200
    checkpoint_interval = 50
    cluster_weight = 0.000000001  # adjustable weight for cluster loss on unknown types
    tau = 1.0  # temperature for Gumbel softmax
    type_embed_dim = 5  # original is 2


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

    # randomization initiate
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = GaussianDataset(A=20, B=4, num_samples=20000, image_size=image_size, is_unknown_center_new=is_unknown_center_new,
                              specific_known_cells=specific_known, num_total_types=num_total_types, num_known_types=num_known_types,
                              boundary=boundary)
    dataset.plot_sample(0, save_folder=savefig_dir, save_name=f'{filename_fixed}_plot_cell_RF.png')
    dataset.plot_sample(1, save_folder=savefig_dir, save_name=f'{filename_fixed}_plot_cell_RF.png')
    dataset.print_cell_table()

    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = CrossAttentionNet(d_model=32, hidden_dim=32, center_B=10, num_total_types=num_total_types, type_embed_dim=type_embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    mse_loss = nn.MSELoss()

    losses_dict = {"epochs": [], "total_loss": [], "reg_loss": [], "cluster_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_total_loss = 0.0
        running_reg_loss = 0.0
        running_cluster_loss = 0.0
        total_samples = 0

        for batch in loader:
            optimizer.zero_grad()

            # Unpack the batch; note that your dataset should now provide the keys below.
            images = batch['image']                     # [B, 1, H, W]
            target = batch['target']                    # [B]
            query_center = batch['query_center']        # [B, 2]
            is_center_known = batch['is_center_known']    # [B] bool
            unknown_center_id = batch['unknown_center_id']# [B] long
            type_gt = batch['type_gt']                    # [B] long
            is_type_known = batch['is_type_known']        # [B] bool
            cell_idx = batch['cell_idx']                  # [B] long


            # Forward pass: returns regression predictions, type logits, and discrete predictions (for unknown samples)
            target_pred, global_entropy = model(images, query_center, is_center_known,
                                                        unknown_center_id, type_gt, is_type_known, cell_idx, tau=tau)

            # 1. Regression loss for the target prediction
            loss_reg = mse_loss(target_pred, target)

            loss_cluster = cluster_weight * global_entropy

            if (epoch + 1) > 100:
              total_loss = loss_reg + loss_cluster
            else:
              total_loss = loss_reg

            total_loss.backward()
            optimizer.step()
            # Clamp the unknown center embedding weights as in the original code
            model.unknown_embedding.weight.data.clamp_(-0.999, 0.999)

            batch_size = images.size(0)
            running_total_loss += total_loss.item() * batch_size
            running_reg_loss += loss_reg.item() * batch_size
            running_cluster_loss += loss_cluster.item() * batch_size
            total_samples += batch_size

        # Step the scheduler after each epoch
        scheduler.step()

        # Compute average losses per sample for the epoch
        epoch_total_loss = running_total_loss / total_samples if total_samples > 0 else 0.0
        epoch_reg_loss = running_reg_loss / total_samples if total_samples > 0 else 0.0
        epoch_cluster_loss = running_cluster_loss / total_samples if total_samples > 0 else 0.0

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {epoch_total_loss:.6f}, "
              f"Reg Loss: {epoch_reg_loss:.6f}, Cluster Loss: {epoch_cluster_loss:.6f} \n")
        # Store losses for this epoch.
        losses_dict["epochs"].append(epoch + 1)
        losses_dict["total_loss"].append(epoch_total_loss)
        losses_dict["reg_loss"].append(epoch_reg_loss)
        losses_dict["cluster_loss"].append(epoch_cluster_loss)

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

    # Assume 'dataset' is your GaussianDataset instance
    # and 'model' is your trained CrossAttentionNet instance.
    # Also assume that dataset has an attribute "type_known_flags" that is a list
    # with a Boolean value for each cell indicating if its type is provided.
    # If not available, we infer known cells as those with target type < dataset.num_known_types.

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
          learned_center_embedding = model.unknown_embedding(torch.tensor(unknown_center_id, dtype=torch.long)).detach().cpu().numpy()
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

            logits = model.cell_type_logits(torch.tensor(unknown_index, dtype=torch.long).to(device))
            learned_embedding = logits.detach().cpu().numpy()
            # logits = self.cell_type_logits(cell_idx[i].long())  # shape: [num_total_types]
            # Use Gumbel-Softmax (with tau=1.0, hard=True) to get a nearly one-hot vector.
            one_hot = F.gumbel_softmax(logits, tau=0.001, hard=True)
            # predicted_type = int(torch.argmax(one_hot).item())
            predicted_type = int(torch.argmax(logits).item())



            # Retrieve the type embedding as the weighted sum over the fixed type embeddings.
            # learned_embedding = (one_hot.unsqueeze(0) @ model.type_embedding.weight).detach().cpu().numpy().flatten()

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

    # Convert DataFrame to a dictionary suitable for MATLAB
    matlab_data = {
        "column_names":np.array(df.columns, dtype=object),  # Store column names
        "data": np.array(df.values.tolist(), dtype=object)    # Convert DataFrame rows into a list of lists
    }

    # folder_path = '/content/drive/MyDrive/Colab/PreyCapture/'
    # saved_mat_path = os.path.join(folder_path, "cells_info.mat")
    # scipy.io.savemat(saved_mat_path, matlab_data)


if __name__ == '__main__':
    main()





