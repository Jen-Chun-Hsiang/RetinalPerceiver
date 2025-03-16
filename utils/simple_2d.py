import torch
import numpy as np
import random
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

##############################
# Model: CNN + Cross-Attention with Type Learning via Gumbel Softmax
##############################

class CrossAttentionNet(nn.Module):
    def __init__(self, d_model=32, hidden_dim=32, center_B=10, num_total_types=5, type_embed_dim=2):
        """
        d_model: transformer embedding dimension.
        hidden_dim: hidden layer dimension.
        center_B: number of unknown center embeddings (should match B in dataset).
        num_total_types: total number of type classes (maximum available types).
        type_embed_dim: dimension for type embeddings.
        """
        super(CrossAttentionNet, self).__init__()
        self.cnn1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.key_proj = nn.Linear(16, d_model)
        self.value_proj = nn.Linear(16, d_model)
        # Query is the concatenation of center (2 dims) and type embedding (type_embed_dim)
        self.query_proj = nn.Linear(2 + type_embed_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, batch_first=False)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Center: learnable embedding for unknown centers (when center is missing)
        self.unknown_embedding = nn.Embedding(center_B, 2)
        # Fixed type embedding table (each row corresponds to a discrete type).
        # We will freeze this embedding so it remains fixed.
        self.type_embedding = nn.Embedding(12, type_embed_dim)
        self.type_embedding.weight.requires_grad = False  # freeze type embedding

        # For unknown type queries, we have a learnable lookup table that produces
        # a continuous logits vector for each cell. These logits (of length num_total_types)
        # are then transformed via Gumbel-Softmax to get a nearly one-hot distribution.
        self.cell_type_logits = nn.Embedding(24, 12)

        pos_encoding = get_2d_sincos_positional_encoding(8, 8, d_model)
        self.register_buffer('positional_encoding', pos_encoding)

    def forward(self, x, query_center, is_center_known, unknown_center_id,
                type_gt, is_type_known, cell_idx, tau=1.0):
        """
        x: input image tensor of shape [batch, 1, H, W]
        query_center: [batch, 2] normalized center coordinates.
        is_center_known: boolean tensor [batch] indicating if center is provided.
        unknown_center_id: tensor [batch] with indices for unknown center embedding.
        type_gt: tensor [batch] ground-truth type ids (used when type info is known).
        is_type_known: boolean tensor [batch] indicating if type info is provided.
        cell_idx: tensor [batch] with cell indices for looking up cell type logits.
        tau: temperature for Gumbel-Softmax.
        """
        # Process image through CNN to get tokens.
        x = F.relu(self.pool1(self.cnn1(x)))
        x = F.relu(self.pool2(self.cnn2(x)))
        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)  # [B, num_tokens, C]
        keys = self.key_proj(tokens) + self.positional_encoding.unsqueeze(0).to(x.device)
        values = self.value_proj(tokens) + self.positional_encoding.unsqueeze(0).to(x.device)

        # Process center: for unknown centers, replace with the corresponding learnable embedding.
        query_center_mod = query_center.clone()
        if (~is_center_known).any():
            unknown_idx = torch.nonzero(~is_center_known).squeeze(1)
            query_center_mod[unknown_idx] = self.unknown_embedding(unknown_center_id[unknown_idx])

        # Process type part:
        # For each cell, if the type is known, use the ground-truth one-hot encoding.
        # Otherwise, use the learnable logits from cell_type_logits, apply Gumbel-Softmax to get
        # a near one-hot distribution, and use that to retrieve the type embedding.
        type_query = []
        unknown_probs_list = []  # store unknown cells' soft one-hot vectors for regularization.
        for i in range(B):
            if is_type_known[i]:
                # Ground-truth: create one-hot vector.
                one_hot = F.one_hot(type_gt[i].long(), num_classes=self.type_embedding.num_embeddings).float()
                # Compute type embedding as a weighted sum of the fixed type embeddings.
                type_emb = torch.matmul(one_hot, self.type_embedding.weight)
                type_query.append(type_emb)
            else:
                # Unknown: look up the logits vector for this cell.
                logits = self.cell_type_logits(cell_idx[i].long())  # shape: [num_total_types]
                # Apply Gumbel-Softmax to get a nearly one-hot distribution.
                one_hot = F.gumbel_softmax(logits, tau=tau, hard=True)
                # Save the distribution for global entropy regularization.
                unknown_probs_list.append(one_hot)
                # Retrieve type embedding.
                type_emb = torch.matmul(one_hot, self.type_embedding.weight)
                type_query.append(type_emb)
        type_query = torch.stack(type_query, dim=0)  # [B, type_embed_dim]

        # Compute global entropy of unknown type assignments for regularization.
        if len(unknown_probs_list) > 0:
            unknown_probs = torch.stack(unknown_probs_list, dim=0)  # [N_unknown, num_total_types]
            avg_prob = unknown_probs.mean(dim=0)
            global_entropy = - torch.sum(avg_prob * torch.log(avg_prob + 1e-10))
        else:
            global_entropy = torch.tensor(0.0, device=x.device)

        # Form the full query by concatenating the center and type parts.
        full_query = torch.cat([query_center_mod, type_query], dim=1)
        q = self.query_proj(full_query).unsqueeze(0)  # [1, B, d_model]

        attn_output, _ = self.attn(q, keys.transpose(0, 1), values.transpose(0, 1))
        attended = attn_output.squeeze(0)  # [B, d_model]

        # Final regression output.
        out = F.relu(self.fc1(attended))
        target_pred = self.fc2(out).squeeze(1)

        # Return the main prediction along with the computed global entropy.
        # The global entropy term can be weighted and added to your main loss externally.
        return target_pred, global_entropy

##############################
# Helper: 2D Sinusoidal Positional Encoding
##############################
def get_2d_sincos_positional_encoding(H, W, d_model):
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for 2D positional encoding")
    d_model_half = d_model // 2

    y_pos = torch.arange(H, dtype=torch.float32).unsqueeze(1)
    x_pos = torch.arange(W, dtype=torch.float32).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model_half, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model_half))

    pe_y = torch.zeros(H, d_model_half)
    pe_y[:, 0::2] = torch.sin(y_pos * div_term)
    pe_y[:, 1::2] = torch.cos(y_pos * div_term)

    pe_x = torch.zeros(W, d_model_half)
    pe_x[:, 0::2] = torch.sin(x_pos * div_term)
    pe_x[:, 1::2] = torch.cos(x_pos * div_term)

    pe_y = pe_y.unsqueeze(1).repeat(1, W, 1)
    pe_x = pe_x.unsqueeze(0).repeat(H, 1, 1)

    pe = torch.cat([pe_y, pe_x], dim=-1).view(H * W, d_model)
    return pe


##############################
# Dataset: Cells with Fixed Gaussians and Type Masking
##############################
class GaussianDataset(Dataset):
    def __init__(self, A=22, B=10, image_size=32, num_samples=1000, num_total_types=5, num_known_types=3,
                 boundary=4, is_unknown_center_new=False, specific_known_cells=None, masked_type_perc = 0.33):
        """
        Creates A known cells and B unknown cells, each with a fixed Gaussian and an assigned type.
        num_total_types: total number of type_ids (e.g., 5). Types are indexed from 0.
        num_known_types: for cells with type in [0, num_known_types-1] the type info is provided with 80% chance.
        For cells with type >= num_known_types, type info is always masked.
        """
        self.image_size = image_size
        self.num_samples = num_samples
        self.is_unknown_center_new = is_unknown_center_new
        self.num_total_types = num_total_types
        self.num_known_types = num_known_types

        # Create grid for PDF computation
        x_coords = np.arange(self.image_size)
        y_coords = np.arange(self.image_size)
        xv, yv = np.meshgrid(x_coords, y_coords, indexing='xy')
        grid = np.stack([xv, yv], axis=-1)  # (image_size, image_size, 2)

        self.cell_properties = []
        self.pdf_tensors = []
        known_centers = []
        type_covs = {}

        num_specific = 0
        if specific_known_cells is not None:
            num_specific = len(specific_known_cells)
            if num_specific > A:
                raise ValueError("More specific known cells provided than defined A.")
            for cell in specific_known_cells:
                center = np.array(cell["center"])
                theta = cell["theta"]
                eig1 = cell["eig1"]
                eig2 = cell["eig2"]
                type_id = cell["type_id"]  # assumed to be in [0, num_total_types-1]
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
                cov = R @ np.diag([eig1, eig2]) @ R.T
                if type_id not in type_covs:
                    type_covs[type_id] = cov

                self.cell_properties.append({
                    "center": center,
                    "cov": cov,
                    "type_id": type_id
                })
                known_centers.append(center)
                pdf_tensor = self._compute_gaussian_pdf(grid, center, cov)
                self.pdf_tensors.append(pdf_tensor)

        # Generate remaining known cells randomly
        remaining_known = A - num_specific

        if remaining_known > 0:
            known_type_ids = np.repeat(np.arange(num_total_types), remaining_known // num_total_types)
            known_type_ids = np.concatenate((known_type_ids, np.random.choice(num_total_types, remaining_known % num_total_types, replace=False)))
            # np.random.shuffle(known_type_ids)
            # known_type_ids = np.random.choice(range(num_total_types), size=remaining_known, replace=True)
            for i in range(remaining_known):
                center = np.random.uniform(boundary, image_size-boundary, size=2)
                type_id = int(known_type_ids[i])
                if type_id in type_covs:
                    cov = type_covs[type_id]
                else:
                    theta = np.random.uniform(0, 2 * np.pi)
                    eig1, eig2 = np.random.uniform(2, 5, size=2)
                    R = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]])
                    cov = R @ np.diag([eig1, eig2]) @ R.T
                    type_covs[type_id] = cov
                self.cell_properties.append({
                    "center": center,
                    "cov": cov,
                    "type_id": type_id
                })
                known_centers.append(center)
                pdf_tensor = self._compute_gaussian_pdf(grid, center, cov)
                self.pdf_tensors.append(pdf_tensor)



        # Unknown cells: for these, we assign types from [num_known_types, num_total_types-1]
        self.A = A
        self.B = B
        for i in range(B):
            if is_unknown_center_new:
                center = np.random.uniform(boundary, image_size-boundary, size=2)
            else:
                center = known_centers[np.random.randint(0, A)]
            unknown_possible = np.arange(num_known_types, num_total_types)
            type_id = int(np.random.choice(unknown_possible))
            theta = np.random.uniform(0, 2 * np.pi)
            eig1, eig2 = np.random.uniform(2, 5, size=2)
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])
            cov = R @ np.diag([eig1, eig2]) @ R.T
            self.cell_properties.append({
                "center": center,
                "cov": cov,
                "type_id": type_id
            })
            pdf_tensor = self._compute_gaussian_pdf(grid, center, cov)
            self.pdf_tensors.append(pdf_tensor)

        self.num_cells = self.A + self.B

        # Precompute the fixed flag for whether type is known for each cell.
        self.masked_type_perc = masked_type_perc
        self.type_known_flags = []
        for cell in self.cell_properties:
            # For types in [0, num_known_types-1] provide with 80% chance; otherwise, always mask.
            if cell["type_id"] < num_known_types:
                self.type_known_flags.append(random.random() > masked_type_perc)
            else:
                self.type_known_flags.append(False)

    def _compute_gaussian_pdf(self, grid, center, cov):
        diff = grid - center
        inv_cov = np.linalg.inv(cov)
        exponent = -0.5 * np.einsum('...i,ij,...j', diff, inv_cov, diff)
        norm_factor = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
        pdf = norm_factor * np.exp(exponent)
        return torch.tensor(pdf, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
          - image: new white noise image.
          - target: dot-product between image and precomputed Gaussian PDF.
          - is_center_known: flag (first A cells are known).
          - query_center: normalized center (or to be replaced if unknown).
          - unknown_center_id: index for unknown center embedding (if applicable).
          - type_gt: ground-truth type id (integer in [0, num_total_types-1]).
          - is_type_known: boolean flag; for types in [0, num_known_types-1] provided with 80% chance,
            for types >= num_known_types, always masked.
          - cell_idx: index of the cell.
        """
        cell_idx = random.randint(0, self.num_cells - 1)
        image = torch.rand(1, self.image_size, self.image_size) * 2 - 1
        pdf_tensor = self.pdf_tensors[cell_idx]
        target = torch.dot(image.view(-1), pdf_tensor.view(-1))

        is_center_known = cell_idx < self.A
        center = self.cell_properties[cell_idx]["center"]
        center_norm = (center / self.image_size) * 2 - 1  # normalized
        unknown_center_id = torch.tensor(cell_idx - self.A, dtype=torch.long) if not is_center_known else torch.tensor(-1, dtype=torch.long)

        type_gt = torch.tensor(self.cell_properties[cell_idx]["type_id"], dtype=torch.long)
        is_type_known = torch.tensor(self.type_known_flags[cell_idx], dtype=torch.bool)

        return {
            'image': image,
            'target': target,
            'is_center_known': torch.tensor(is_center_known, dtype=torch.bool),
            'query_center': torch.tensor(center_norm, dtype=torch.float32),
            'unknown_center_id': unknown_center_id,
            'type_gt': type_gt,
            'is_type_known': is_type_known,
            'cell_idx': cell_idx
        }

    def plot_sample(self, index=None, save_folder=None, save_name=None):
        if index is None:
            index = random.randint(0, self.num_cells - 1)
        pdf_tensor = self.pdf_tensors[index].numpy()
        noise_image = torch.randn(1, self.image_size, self.image_size).squeeze(0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sns.heatmap(pdf_tensor, cmap="viridis", ax=axes[0])
        axes[0].set_title(f"Gaussian PDF Heatmap (Cell {index})")
        axes[0].invert_yaxis()

        axes[1].imshow(noise_image, cmap="gray")
        axes[1].set_title("Random Noise Image")
        plt.tight_layout()
        if save_folder is not None:
            filepath = os.path.join(save_folder, f"{index}_{save_name}")
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def print_cell_table(self):
        data = []
        for idx, cell in enumerate(self.cell_properties):
            center = cell["center"]
            cov = cell["cov"]
            row = {
                "Cell ID": idx,
                "Type ID": cell["type_id"],
                "is_type_known": self.type_known_flags[idx],
                "is_center_known" : idx < self.A,
                "Center X": center[0],
                "Center Y": center[1],
                "Cov_00": cov[0, 0],
                "Cov_01": cov[0, 1],
                "Cov_10": cov[1, 0],
                "Cov_11": cov[1, 1],
            }
            data.append(row)
        df = pd.DataFrame(data)
        logging.info(f'df: {df} \n')