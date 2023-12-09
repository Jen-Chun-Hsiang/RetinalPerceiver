import numpy as np
import torch

class TargetMatrixGenerator:
    def __init__(self, mean=(0, 0), cov=((1, 0), (0, 1)), device='cpu'):
        self.mean = mean
        self.cov = cov
        self.device = device

    def create_3d_target_matrix(self, input_height, input_width, input_depth):
        # Define a Gaussian function
        def gaussian(x, sigma, mean_t):
            return np.exp(-np.power(x - mean_t, 2.) / (2 * np.power(sigma, 2.)))

        # Define the function Ypf with T as input
        def Ypf(T, w):
            gauss1 = gaussian(T, w[0], w[2]) * w[4]
            gauss2 = gaussian(T, w[1], w[3]) * w[5]
            return (gauss1 - gauss2) + w[6]

        SamplingRate = 40
        T = np.arange(-1, 3, 1 / SamplingRate)
        # Calculate freqf_t
        T_positive = T[(T >= 0) & (T < 0.5)]
        freqf_t = Ypf(T_positive, [0.05, 0.12, 0.08, 0.12, 1, 0.0, 0])
        target_matrix = np.array([self.generate_2d_gaussian((input_width, input_height)) * time_point for time_point in freqf_t[:input_depth]])
        return torch.tensor(target_matrix, dtype=torch.float32).to(self.device)

    def generate_2d_gaussian(self, size):
        x = np.linspace(-1, 1, size[0])
        y = np.linspace(-1, 1, size[1])
        x, y = np.meshgrid(x, y)

        d = np.dstack([x, y])
        gaussian_matrix = np.exp(-0.5 * np.sum(np.dot(d - self.mean, np.linalg.inv(self.cov)) * (d - self.mean), axis=2))

        return gaussian_matrix
