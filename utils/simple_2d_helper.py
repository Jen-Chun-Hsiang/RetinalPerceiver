import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os
import numpy as np
import random
import logging
import pandas as pd
from scipy.io import savemat
from torch.utils.data import Dataset
import os
import torch
import numpy as np


def adaptive_grad_clip(parameters, clip_factor=0.01, eps=1e-3):
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = torch.norm(param.detach())
        grad_norm = torch.norm(param.grad.detach())

        threshold = param_norm * clip_factor + eps

        if grad_norm > threshold:
            param.grad.data.mul_(threshold / (grad_norm + 1e-6))


class SharedPerturbationOptimizer:
    def __init__(self, model, image, query1, query2, target1, target2,
                 lr=1e-3, max_iter=300, tv_weight=0.5, tolerance=1e-6,
                 batch_size=256, noisy_image_scaling=1.0, directional_loss_weight=0.0):
        """
        Optimizes a shared perturbation image (initialized as blank) that is subtracted from a
        batch of random noise images. The model is then fed these perturbed images for two queries,
        and the optimizer updates the shared perturbation to make the model outputs approach
        their respective target values.

        Parameters:
        - model: The trained neural network model. It is assumed to be already on the correct device.
        - image: A template image tensor of shape (1, H, W) whose dimensions will be used for the
                 shared perturbation image. (Already on the correct device.)
        - query1, query2: The two query inputs to the model (already on the correct device).
        - target1, target2: The desired output values for query1 and query2 (already on the correct device).
        - lr: Learning rate for the optimizer.
        - max_iter: Maximum number of optimization iterations.
        - tv_weight: Weight for total variation loss (regularizes smoothness).
        - tolerance: Threshold for early stopping (not used explicitly here).
        - batch_size: Number of random noise images generated per iteration.
        - noisy_image_scaling: Scaling factor for the noise.
        - directional_loss_weight: Weight for the directional loss that encourages
          (output2 - output1) to match (target2 - target1). Set to 0 to disable.
        """
        self.model = model.eval()  # Model is assumed to be on the correct device.
        for param in self.model.parameters():
            param.requires_grad = False

        self.query1 = query1
        self.query2 = query2
        self.target1 = target1
        self.target2 = target2
        self.max_iter = max_iter
        self.tv_weight = tv_weight
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.noisy_image_scaling = noisy_image_scaling
        self.directional_loss_weight = directional_loss_weight

        # Infer device from the image tensor.
        self.device = image.device

        # Initialize the shared perturbation image as a blank image using the provided image's shape.
        if image.dim() == 4 and image.shape[0] == 1:
            self.orig_image = torch.zeros_like(image[0:1])
        else:
            self.orig_image = torch.zeros_like(image)
        self.orig_image = self.orig_image.to(self.device)
        self.shared_perturbation = self.orig_image.clone().detach()
        self.shared_perturbation.requires_grad = True

        self.optimizer = optim.Adam([self.shared_perturbation], lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=4, T_mult=2, eta_min=1e-6)

    def total_variation_loss(self, x, x_orig):
        """Compute total variation loss to encourage smooth perturbations."""
        delta_x = x - x_orig
        tv_h = torch.abs(delta_x[:, :, 1:] - delta_x[:, :, :-1]).sum()
        tv_w = torch.abs(delta_x[:, 1:, :] - delta_x[:, :-1, :]).sum()
        return (tv_h + tv_w) / delta_x.numel()

    def perturbation_loss_l1(self, x, x_orig, weight=1e-1):
        """L1 regularization loss on the perturbation."""
        return weight * torch.mean(torch.abs(x - x_orig))

    def perturbation_loss_l2(self, x, x_orig, weight=5e-4):
        """L2 regularization loss on the perturbation."""
        return weight * (torch.norm(x - x_orig, p=2) ** 2)

    def closure(self):
        """Performs a forward pass with a new random noise batch and computes the total loss."""
        self.optimizer.zero_grad()
        device = self.device  # Use the inferred device.

        # Generate random noise batch.
        if self.shared_perturbation.dim() == 4:
            _, channels, height, width = self.shared_perturbation.shape
        else:
            channels, height, width = self.shared_perturbation.shape
        noise_batch = 2 * torch.rand((self.batch_size, channels, height, width), device=device) - 1
        noise_batch = self.noisy_image_scaling * noise_batch

        # Create perturbed batch.
        perturbed_batch = torch.clamp(noise_batch - self.shared_perturbation, -1, 1)

        # Prepare extra inputs.
        bool_tensor = torch.tensor([True], device=device).expand(self.batch_size, 1)
        num_tensor = torch.tensor([-1.0], device=device).expand(self.batch_size, 1)

        query1_batch = self.query1.expand(self.batch_size, -1)
        query2_batch = self.query2.expand(self.batch_size, -1)

        # Forward pass.
        output1 = self.model(perturbed_batch, query1_batch, bool_tensor, num_tensor)
        output2 = self.model(perturbed_batch, query2_batch, bool_tensor, num_tensor)

        loss1 = F.mse_loss(output1, self.target1.view(1).expand_as(output1))
        loss2 = F.mse_loss(output2, self.target2.view(1).expand_as(output2))

        tv_loss = self.tv_weight * self.total_variation_loss(self.shared_perturbation, self.orig_image)
        pert_loss_l1 = self.perturbation_loss_l1(self.shared_perturbation, self.orig_image)
        pert_loss_l2 = self.perturbation_loss_l2(self.shared_perturbation, self.orig_image)

        if self.directional_loss_weight > 0:
            directional_target = self.target2 - self.target1
            directional_target_expanded = directional_target.view(1).expand_as(output2 - output1)
            directional_loss = F.mse_loss(output2 - output1, directional_target_expanded)
            total_loss = loss1 + loss2 + directional_loss * self.directional_loss_weight + tv_loss + pert_loss_l1 + pert_loss_l2
        else:
            total_loss = loss1 + loss2 + tv_loss + pert_loss_l1 + pert_loss_l2

        total_loss.backward()

        if self.directional_loss_weight > 0:
            self.final_losses = {
                "Prediction Loss (Query 1)": loss1.item(),
                "Prediction Loss (Query 2)": loss2.item(),
                "Directional Loss": directional_loss.item(),
                "L1 Regularization Loss": pert_loss_l1.item(),
                "L2 Regularization Loss": pert_loss_l2.item(),
                "Total Variation Loss": tv_loss.item(),
                "Total Loss": total_loss.item()
            }
        else:
            self.final_losses = {
                "Prediction Loss (Query 1)": loss1.item(),
                "Prediction Loss (Query 2)": loss2.item(),
                "L1 Regularization Loss": pert_loss_l1.item(),
                "L2 Regularization Loss": pert_loss_l2.item(),
                "Total Variation Loss": tv_loss.item(),
                "Total Loss": total_loss.item()
            }
        return total_loss

    def optimize(self):
        """Runs the optimization loop, updating the shared perturbation image."""
        for i in range(self.max_iter):
            self.optimizer.zero_grad()
            loss = self.closure()
            self.optimizer.step()
            self.shared_perturbation.data.clamp_(-1, 1)
            self.scheduler.step()

            if i % 10 == 0:
                logging.info("Iteration %d: Total Loss = %.6f", i, loss.item())

        return (self.orig_image - self.shared_perturbation).detach()

    def print_model_outputs(self):
        """Logs the model output values for each query after optimization."""
        with torch.no_grad():
            final_image = self.orig_image - self.shared_perturbation
            logging.info("Final Image Shape: %s", final_image.shape)
            bool_tensor = torch.tensor([[True]], device=self.device)
            num_tensor = torch.tensor([[-1.0]], device=self.device)

            output1 = self.model(final_image, self.query1, bool_tensor, num_tensor)
            output2 = self.model(final_image, self.query2, bool_tensor, num_tensor)

        logging.info("Final Model Output for Query 1: %s", output1.cpu().numpy())
        logging.info("Final Model Output for Query 2: %s", output2.cpu().numpy())

    def print_final_losses(self):
        """Logs final loss values recorded during the last optimization iteration."""
        logging.info("Final Loss Breakdown:")
        logging.info("Prediction Loss (Query 1): %.6f", self.final_losses.get("Prediction Loss (Query 1)", 0))
        logging.info("Prediction Loss (Query 2): %.6f", self.final_losses.get("Prediction Loss (Query 2)", 0))
        logging.info("Perturbation Loss L1: %.6f", self.final_losses.get("L1 Regularization Loss", 0))
        logging.info("Perturbation Loss L2: %.6f", self.final_losses.get("L2 Regularization Loss", 0))
        if self.directional_loss_weight > 0:
            logging.info("Directional Loss L2: %.6f", self.final_losses.get("Directional Loss", 0))
        logging.info("Total Variation Loss: %.6f", self.final_losses.get("Total Variation Loss", 0))
        logging.info("Total Loss: %.6f", self.final_losses.get("Total Loss", 0))

    def evaluate_model(self):
        """
        Compares the model's predictions on the original blank image with those on the optimized image.
        This helps determine if the optimization is effectively driving the outputs closer to the targets.
        """
        with torch.no_grad():
            orig_batch = self.orig_image.unsqueeze(0).expand(self.batch_size, -1, -1, -1)
            final_image = self.orig_image - self.shared_perturbation
            final_batch = final_image.unsqueeze(0).expand(self.batch_size, -1, -1, -1)
            bool_tensor = torch.tensor([True]).expand(self.batch_size, 1).to(self.device)
            num_tensor = torch.tensor([-1.0]).expand(self.batch_size, 1).to(self.device)

            orig_output1 = self.model(orig_batch, self.query1, bool_tensor, num_tensor)
            orig_output2 = self.model(orig_batch, self.query2, bool_tensor, num_tensor)
            opt_output1 = self.model(final_batch, self.query1, bool_tensor, num_tensor)
            opt_output2 = self.model(final_batch, self.query2, bool_tensor, num_tensor)

            orig_loss1 = F.mse_loss(orig_output1, self.target1.view_as(orig_output1))
            orig_loss2 = F.mse_loss(orig_output2, self.target2.view_as(orig_output2))
            opt_loss1 = F.mse_loss(opt_output1, self.target1.view_as(opt_output1))
            opt_loss2 = F.mse_loss(opt_output2, self.target2.view_as(opt_output2))

        logging.info("Evaluation Report:")
        logging.info("Before Optimization (Blank Image):")
        logging.info("  Query 1 Loss: %.6f", orig_loss1.item())
        logging.info("  Query 2 Loss: %.6f", orig_loss2.item())
        logging.info("After Optimization (Blank - Perturbation):")
        logging.info("  Query 1 Loss: %.6f", opt_loss1.item())
        logging.info("  Query 2 Loss: %.6f", opt_loss2.item())

        improvement1 = orig_loss1.item() - opt_loss1.item()
        improvement2 = orig_loss2.item() - opt_loss2.item()
        logging.info("Improvements:")
        logging.info("  Query 1 Loss Reduction: %.6f", improvement1)
        logging.info("  Query 2 Loss Reduction: %.6f", improvement2)


import os
import numpy as np
import torch
import random
import logging
import pandas as pd
from scipy.io import savemat
from torch.utils.data import Dataset


class ComplementaryGaussianDataset(Dataset):
    def __init__(self, current_dataset, max_new_cells_per_type=16, random_center=False, selected_types=None):
        """
        Create a complementary dataset from an existing GaussianDataset instance.

        Parameters:
          - current_dataset: an instance of GaussianDataset containing the original cell properties.
          - max_new_cells_per_type: maximum number of new cells to generate for each selected type (default: 16).
          - random_center: if False (default), new cells are generated by permuting centers from the current dataset.
                           If True, centers are generated randomly, bounded by the min and max coordinates of current cells.
          - selected_types: a list of type_ids to generate new cells for. If None, use all types found in current_dataset.

        Each new cell will have the same type-specific parameters as in the current dataset. However, we ensure that
        the (center, type) combination is not present in the current dataset.
        """
        self.image_size = current_dataset.image_size
        x_coords = np.arange(self.image_size)
        y_coords = np.arange(self.image_size)
        xv, yv = np.meshgrid(x_coords, y_coords, indexing='xy')
        self.grid = np.stack([xv, yv], axis=-1)  # shape: (image_size, image_size, 2)

        # Extract cell properties from the current dataset.
        current_cells = current_dataset.cell_properties

        # For each type, store the first encountered parameter set.
        self.type_params = {}
        for cell in current_cells:
            t = cell["type_id"]
            if t not in self.type_params:
                self.type_params[t] = {
                    "theta": cell["theta"],
                    "eig1": cell["eig1"],
                    "eig2": cell["eig2"],
                    "stretching_factor": cell["stretching_factor"],
                    "surround_strength": cell["surround_strength"]
                }

        # Determine which types to generate new cells for.
        if selected_types is not None:
            self.types_to_generate = selected_types
        else:
            self.types_to_generate = list(self.type_params.keys())

        # Build a lookup of centers already used in the current dataset for each type.
        # We first extract unique centers from current_cells.
        unique_centers = np.unique(np.array([cell["center"] for cell in current_cells]), axis=0)
        self.unique_centers = unique_centers  # for permutation option

        # For each type in self.types_to_generate, collect unique centers (as in current dataset) that have that type.
        self.current_centers_by_type = {t: [] for t in self.types_to_generate}
        for cell in current_cells:
            t = cell["type_id"]
            if t in self.current_centers_by_type:
                self.current_centers_by_type[t].append(cell["center"])
        # Convert each to a unique numpy array.
        for t in self.current_centers_by_type:
            if len(self.current_centers_by_type[t]) > 0:
                self.current_centers_by_type[t] = np.unique(np.array(self.current_centers_by_type[t]), axis=0)
            else:
                self.current_centers_by_type[t] = np.array([])

        # If using random centers, compute overall bounds from all centers in current dataset.
        all_centers = np.array([cell["center"] for cell in current_cells])
        self.x_min, self.x_max = np.min(all_centers[:, 0]), np.max(all_centers[:, 0])
        self.y_min, self.y_max = np.min(all_centers[:, 1]), np.max(all_centers[:, 1])

        # Create new cell properties and compute their PDF tensors.
        self.cell_properties = []
        self.pdf_tensors = []

        if not random_center:
            # Permutation option: use unique centers from current dataset.
            for t in self.types_to_generate:
                count = 0
                current_centers = self.current_centers_by_type[t]
                # For each candidate center, add it only if (center, t) does not exist in current dataset.
                for center in self.unique_centers:
                    if current_centers.size > 0 and any(np.allclose(center, c) for c in current_centers):
                        continue
                    params = self.type_params[t]
                    new_cell = {
                        "center": center,
                        "theta": params["theta"],
                        "eig1": params["eig1"],
                        "eig2": params["eig2"],
                        "stretching_factor": params["stretching_factor"],
                        "surround_strength": params["surround_strength"],
                        "type_id": t
                    }
                    self.cell_properties.append(new_cell)
                    pdf_tensor = self._compute_gaussian_pdf(self.grid, center,
                                                            params["theta"], params["eig1"], params["eig2"],
                                                            params["stretching_factor"], params["surround_strength"])
                    self.pdf_tensors.append(pdf_tensor)
                    count += 1
                    if count >= max_new_cells_per_type:
                        break
        else:
            # Random center option: generate new centers within bounds that are unique for each type.
            for t in self.types_to_generate:
                count = 0
                current_centers = self.current_centers_by_type[t]
                # Keep track of centers already generated for this type in the complementary dataset.
                generated_centers = []
                while count < max_new_cells_per_type:
                    new_center = np.array([
                        np.random.uniform(self.x_min, self.x_max),
                        np.random.uniform(self.y_min, self.y_max)
                    ])
                    # Check if new_center is already in the current dataset or has been generated already.
                    if (current_centers.size > 0 and any(np.allclose(new_center, c) for c in current_centers)) or \
                            any(np.allclose(new_center, c) for c in generated_centers):
                        continue
                    params = self.type_params[t]
                    new_cell = {
                        "center": new_center,
                        "theta": params["theta"],
                        "eig1": params["eig1"],
                        "eig2": params["eig2"],
                        "stretching_factor": params["stretching_factor"],
                        "surround_strength": params["surround_strength"],
                        "type_id": t
                    }
                    self.cell_properties.append(new_cell)
                    pdf_tensor = self._compute_gaussian_pdf(self.grid, new_center,
                                                            params["theta"], params["eig1"], params["eig2"],
                                                            params["stretching_factor"], params["surround_strength"])
                    self.pdf_tensors.append(pdf_tensor)
                    generated_centers.append(new_center)
                    count += 1

        self.num_cells = len(self.cell_properties)

    def _compute_gaussian_pdf(self, grid, center, theta, eig1, eig2, stretching_factor, surround_strength):
        # Compute rotation matrix based on theta.
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        # Center Gaussian covariance and PDF.
        cov_center = R @ np.diag([eig1, eig2]) @ R.T
        norm_factor_center = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov_center)))
        diff = grid - center
        inv_cov_center = np.linalg.inv(cov_center)
        exponent_center = -0.5 * np.einsum('...i,ij,...j', diff, inv_cov_center, diff)
        center_pdf = norm_factor_center * np.exp(exponent_center)

        # Surround Gaussian: same theta but with stretched eigenvalues.
        eig1_surround = eig1 * stretching_factor
        eig2_surround = eig2 * stretching_factor
        cov_surround = R @ np.diag([eig1_surround, eig2_surround]) @ R.T
        norm_factor_surround = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov_surround)))
        inv_cov_surround = np.linalg.inv(cov_surround)
        exponent_surround = -0.5 * np.einsum('...i,ij,...j', diff, inv_cov_surround, diff)
        surround_pdf = norm_factor_surround * np.exp(exponent_surround)

        # Differential Gaussian: center minus weighted surround.
        pdf = center_pdf - surround_strength * surround_pdf
        return torch.tensor(pdf, dtype=torch.float32)

    def __len__(self):
        return self.num_cells

    def __getitem__(self, idx):
        # For each sample, create a white noise image and compute the dot-product target with the precomputed PDF.
        cell_idx = random.randint(0, self.num_cells - 1)
        image = torch.rand(1, self.image_size, self.image_size) * 2 - 1  # white noise image
        pdf_tensor = self.pdf_tensors[cell_idx]
        target = torch.dot(image.view(-1), pdf_tensor.view(-1))

        # Normalize center to [-1, 1]
        center = self.cell_properties[cell_idx]["center"]
        center_norm = (center / self.image_size) * 2 - 1
        type_gt = torch.tensor(self.cell_properties[cell_idx]["type_id"], dtype=torch.long)
        query = torch.tensor(np.append(center_norm, type_gt), dtype=torch.float32)

        return {
            'image': image,
            'target': target,
            'query_center': torch.tensor(center_norm, dtype=torch.float32),
            'query': query,
            'type_gt': type_gt,
            'cell_idx': cell_idx
        }

    def save_pdf_mat(self, index, save_folder, save_name):
        """
        Save the PDF tensor for a generated cell (by index in this complementary dataset) into a .mat file.
        """
        pdf_tensor = self.pdf_tensors[index].numpy()
        data_dict = {'pdf_tensor': pdf_tensor}
        filename = os.path.join(save_folder, f"{save_name}_{index}_pdf.mat")
        savemat(filename, data_dict)
        logging.info(f"Saved PDF matrix to {filename}")

    def print_cell_table(self, is_shorter=True):
        """
        Print a table summarizing the cell properties for the new cells generated in this complementary dataset.
        """
        data = []
        for idx, cell in enumerate(self.cell_properties):
            center = cell["center"]
            if is_shorter:
                row = {
                    "Cell ID": idx,
                    "Type ID": cell["type_id"],
                    "Center X": center[0],
                    "Center Y": center[1]
                }
            else:
                row = {
                    "Cell ID": idx,
                    "Type ID": cell["type_id"],
                    "Center X": center[0],
                    "Center Y": center[1],
                    "Theta": cell["theta"],
                    "Eig1": cell["eig1"],
                    "Eig2": cell["eig2"],
                    "Stretching Factor": cell["stretching_factor"],
                    "Surround Strength": cell["surround_strength"],
                }
            data.append(row)
        df = pd.DataFrame(data)
        logging.info(f"Complementary Dataset Cell Table:\n{df}\n")


def compute_prediction_error(model, dataset, cell_id, num_stimuli=10000, error_metric='mse', device='cpu'):
    """
    Computes the prediction error for a specific cell by comparing the model's predictions
    with the ground-truth targets computed from the cell's Gaussian PDF.

    Parameters:
      - model: the trained model (e.g., CrossAttentionNet).
      - dataset: instance of GaussianDataset or ComplementaryGaussianDataset.
      - cell_id: integer specifying the cell index.
      - num_stimuli: number of white-noise stimuli to generate.
      - error_metric: error metric to use ('mse' for Mean Squared Error or 'mae' for Mean Absolute Error).
      - device: computation device (e.g., 'cpu' or 'cuda').

    Returns:
      - error: the computed prediction error (a scalar value).
      - outputs: the model outputs (predicted responses) for all stimuli (numpy array).
      - targets: the ground-truth targets for all stimuli (numpy array).
    """
    image_size = dataset.image_size

    # Generate white noise stimuli on the specified device.
    stimuli = torch.rand(num_stimuli, 1, image_size, image_size, device=device) * 2 - 1

    # Retrieve the precomputed PDF tensor for the given cell and ensure it is on device.
    pdf_tensor = dataset.pdf_tensors[cell_id].to(device)
    # Compute ground-truth targets via dot-product (one per stimulus).
    targets = torch.sum(stimuli.view(num_stimuli, -1) * pdf_tensor.view(-1), dim=1)

    # Prepare query components based on dataset type.
    if hasattr(dataset, 'A'):
        # For the current dataset (GaussianDataset): known vs. unknown.
        if cell_id < dataset.A:
            # Known cell: use stored center and type.
            cell = dataset.cell_properties[cell_id]
            center = np.array(cell["center"])
            center_norm = (center / image_size) * 2 - 1  # normalized center
            type_id = cell["type_id"]
            query_center = torch.tensor(center_norm, dtype=torch.float32, device=device).unsqueeze(0).repeat(
                num_stimuli, 1)
            is_center_known = torch.ones(num_stimuli, dtype=torch.bool, device=device)
            unknown_center_id = torch.full((num_stimuli,), -1, dtype=torch.long, device=device)
            type_gt = torch.full((num_stimuli,), type_id, dtype=torch.long, device=device)
            is_type_known = torch.ones(num_stimuli, dtype=torch.bool, device=device)
        else:
            # Unknown cell: use dummy center query and mark type as masked.
            cell = dataset.cell_properties[cell_id]
            center = np.array(cell["center"])
            center_norm = (center / image_size) * 2 - 1
            query_center = torch.tensor(center_norm, dtype=torch.float32, device=device).unsqueeze(0).repeat(
                num_stimuli, 1)
            is_center_known = torch.zeros(num_stimuli, dtype=torch.bool, device=device)
            unknown_center_id = torch.full((num_stimuli,), cell_id - dataset.A, dtype=torch.long, device=device)
            type_id = cell["type_id"]
            type_gt = torch.full((num_stimuli,), type_id, dtype=torch.long, device=device)
            is_type_known = torch.zeros(num_stimuli, dtype=torch.bool, device=device)
    else:
        # For the complementary dataset, treat all cells as "unseen".
        cell = dataset.cell_properties[cell_id]
        center = np.array(cell["center"])
        center_norm = (center / image_size) * 2 - 1
        query_center = torch.tensor(center_norm, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_stimuli, 1)
        is_center_known = torch.zeros(num_stimuli, dtype=torch.bool, device=device)
        # Use cell_id as the unknown center index.
        unknown_center_id = torch.full((num_stimuli,), cell_id, dtype=torch.long, device=device)
        type_id = cell["type_id"]
        type_gt = torch.full((num_stimuli,), type_id, dtype=torch.long, device=device)
        is_type_known = torch.zeros(num_stimuli, dtype=torch.bool, device=device)

    # Create a tensor for the cell index (same value repeated).
    cell_idx = torch.full((num_stimuli,), cell_id, dtype=torch.long, device=device)

    # Move model to device, set evaluation mode, and compute model predictions.
    model.to(device)
    model.eval()
    with torch.no_grad():
        target_pred, global_entropy = model(
            stimuli, query_center, is_center_known, unknown_center_id,
            type_gt, is_type_known, cell_idx, tau=1.0
        )
    outputs = target_pred  # shape: (num_stimuli,)

    # Compute the error metric.
    if error_metric.lower() == 'mse':
        error = torch.mean((outputs - targets) ** 2)
    elif error_metric.lower() == 'mae':
        error = torch.mean(torch.abs(outputs - targets))
    else:
        raise ValueError("Unsupported error metric. Use 'mse' or 'mae'.")

    return error.item(), outputs.cpu().numpy(), targets.cpu().numpy()


def compute_prediction_errors_all(model, dataset, num_stimuli=10000, error_metric='mse', device='cpu'):
    """
    Computes the prediction error for each cell in the given dataset.

    Parameters:
      - model: the trained model.
      - dataset: instance of GaussianDataset or ComplementaryGaussianDataset.
      - num_stimuli: number of white-noise stimuli to generate for each cell.
      - error_metric: error metric to use ('mse' or 'mae').
      - device: computation device (e.g., 'cpu' or 'cuda').

    Returns:
      - errors: a NumPy vector (1D array) of error values, one per cell in the dataset.
      - outputs_all: a list of model output arrays for each cell.
      - targets_all: a list of ground-truth target arrays for each cell.
    """
    # Determine the number of cells from the cell_properties list.
    num_cells = len(dataset.cell_properties)
    errors = np.zeros(num_cells)
    outputs_all = []
    targets_all = []

    # Loop over each cell and compute its prediction error.
    for cell_id in range(num_cells):
        error, outputs, targets = compute_prediction_error(
            model, dataset, cell_id, num_stimuli=num_stimuli, error_metric=error_metric, device=device
        )
        errors[cell_id] = error
        outputs_all.append(outputs)
        targets_all.append(targets)

    return errors, outputs_all, targets_all






