import torch
import numpy as np


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