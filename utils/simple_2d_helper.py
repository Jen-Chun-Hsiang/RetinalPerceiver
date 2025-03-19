import torch


def adaptive_grad_clip(parameters, clip_factor=0.01, eps=1e-3):
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = torch.norm(param.detach())
        grad_norm = torch.norm(param.grad.detach())

        threshold = param_norm * clip_factor + eps

        if grad_norm > threshold:
            param.grad.data.mul_(threshold / (grad_norm + 1e-6))
