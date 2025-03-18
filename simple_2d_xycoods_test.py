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
from utils.simple_2d import GaussianDataset, CrossAttentionNet, compute_sta

import pandas as pd
import scipy.io


def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')
    parser.add_argument('--is_GPU', action='store_true', help='Using GPUs for accelaration')

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
    num_A = 20
    num_B = 4
    seed = 48
    is_unknown_center_new = False
    image_size = 32
    num_total_types = 5
    num_known_types = 3
    boundary = 4
    num_epochs = 200
    checkpoint_interval = 50
    output_mode = 'B'

    # Folders
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    savefig_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'

    os.makedirs(saveprint_dir, exist_ok=True)  # Ensure folder exists
    os.makedirs(savefig_dir, exist_ok=True)  # Ensure folder exists
    os.makedirs(savemodel_dir, exist_ok=True)  # Ensure folder exists
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')

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

    dataset = GaussianDataset(A=num_A, B=num_B, num_samples=20000, image_size=image_size,
                              is_unknown_center_new=is_unknown_center_new,
                              specific_known_cells=specific_known, num_total_types=num_total_types,
                              num_known_types=num_known_types,
                              boundary=boundary, output_mode=output_mode)
    for i in range(5):
        dataset.plot_sample(i, save_folder=savefig_dir, save_name=f'{filename_fixed}_plot_cell_RF.png')
    dataset.print_cell_table()

if __name__ == '__main__':
    main()
