import torch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging


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
                 batch_size=256, noisy_image_scaling=1.0, directional_loss_weight=0.0,
                 device='cuda'):
        """
        Optimizes a shared perturbation image (initialized as blank) that is subtracted from a
        batch of random noise images. The model is then fed these perturbed images for two queries,
        and the optimizer updates the shared perturbation to make the model outputs approach
        their respective target values.

        Parameters:
        - model: The trained neural network model.
        - image: A template image tensor of shape (1, H, W) (single channel) whose dimensions
                 will be used for the shared perturbation image.
        - query1, query2: The two query inputs to the model.
        - target1, target2: The desired output values for query1 and query2.
        - lr: Learning rate for the optimizer.
        - max_iter: Maximum number of optimization iterations.
        - tv_weight: Weight for total variation loss (regularizes smoothness).
        - tolerance: Threshold for early stopping (not used explicitly here).
        - batch_size: Number of random noise images generated per iteration.
        - noisy_image_scaling: Scaling factor for the noise.
        - directional_loss_weight: Weight for the directional loss that encourages
          (output2 - output1) to match (target2 - target1). Set to 0 to disable.
        - device: 'cuda' or 'cpu'.
        """
        self.model = model.to(device).eval()
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.query1 = query1.to(device)
        self.query2 = query2.to(device)
        self.target1 = target1.to(device)
        self.target2 = target2.to(device)
        self.device = device
        self.max_iter = max_iter
        self.tv_weight = tv_weight
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.noisy_image_scaling = noisy_image_scaling
        self.directional_loss_weight = directional_loss_weight

        # Initialize the shared perturbation image as a blank (all zeros) image
        # using the provided image's shape as a template.
        if image.dim() == 4 and image.shape[0] == 1:
            self.orig_image = torch.zeros_like(image[0:1]).to(device)
        else:
            self.orig_image = torch.zeros_like(image).to(device)
        self.shared_perturbation = self.orig_image.clone().detach()
        self.shared_perturbation.requires_grad = True

        # Optimizer and scheduler for the shared perturbation image
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
        l1_loss = torch.mean(torch.abs(x - x_orig))
        return weight * l1_loss

    def perturbation_loss_l2(self, x, x_orig, weight=5e-4):
        """L2 regularization loss on the perturbation."""
        l2_loss = torch.norm(x - x_orig, p=2) ** 2
        return weight * l2_loss

    def closure(self):
        """Performs a forward pass with a new random noise batch and computes the total loss."""
        self.optimizer.zero_grad()

        # Determine the shape for the noise batch.
        if self.shared_perturbation.dim() == 4:
            _, channels, height, width = self.shared_perturbation.shape
        else:
            channels, height, width = self.shared_perturbation.shape
        noise_batch = 2 * torch.rand((self.batch_size, channels, height, width), device=self.device) - 1
        noise_batch = self.noisy_image_scaling * noise_batch

        # Subtract the shared perturbation from each noise image and clip the result to (-1, 1)
        perturbed_batch = noise_batch - self.shared_perturbation
        perturbed_batch = torch.clamp(perturbed_batch, -1, 1)

        # Prepare extra inputs for the model
        bool_tensor = torch.tensor([True]).expand(self.batch_size, 1).to(self.device)
        num_tensor = torch.tensor([-1.0]).expand(self.batch_size, 1).to(self.device)

        query1_batch = self.query1.expand(self.batch_size, -1).to(self.device)
        query2_batch = self.query2.expand(self.batch_size, -1).to(self.device)

        # Forward pass for both queries on the perturbed images
        output1 = self.model(perturbed_batch, query1_batch, bool_tensor, num_tensor)
        output2 = self.model(perturbed_batch, query2_batch, bool_tensor, num_tensor)

        loss1 = F.mse_loss(output1, self.target1.view(1).expand_as(output1))
        loss2 = F.mse_loss(output2, self.target2.view(1).expand_as(output2))

        # Compute regularization losses on the shared perturbation image
        tv_loss = self.tv_weight * self.total_variation_loss(self.shared_perturbation, self.orig_image)
        pert_loss_l1 = self.perturbation_loss_l1(self.shared_perturbation, self.orig_image)
        pert_loss_l2 = self.perturbation_loss_l2(self.shared_perturbation, self.orig_image)

        # Compute the directional loss if enabled
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
            # Ensure the shared perturbation stays within (-1, 1)
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
            num_tensor = torch.tensor([[-1.0]]).to(self.device)

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



