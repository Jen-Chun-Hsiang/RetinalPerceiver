import numpy as np
import torch


class TargetMatrixGenerator:
    def __init__(self, mean=(0, 0), cov=((1, 0), (0, 1)), mean2=None, cov2=None, surround_weight=0.2, device='cpu'):
        self.mean1 = mean
        self.cov1 = cov
        self.mean2 = mean if mean2 is None else mean2
        self.cov2 = cov2
        self.surround_weight = surround_weight
        self.device = device

    def create_3d_target_matrix(self, input_height, input_width, input_depth, tf_surround_weight):
        def gaussian(x, sigma, mean_t):
            return np.exp(-np.power(x - mean_t, 2.) / (2 * np.power(sigma, 2.)))

        def Ypf(T, w):
            gauss1 = gaussian(T, w[0], w[2]) * w[4]
            gauss2 = gaussian(T, w[1], w[3]) * w[5]
            return (gauss1 - gauss2) + w[6]

        SamplingRate = 40
        T = np.arange(-1, 3, 1 / SamplingRate)
        T_positive = T[(T >= 0) & (T < 0.5)]
        freqf_t = Ypf(T_positive, [0.05, 0.12, 0.08, 0.12, 1, tf_surround_weight, 0])

        if self.cov2 is None:
            # Only use the first Gaussian
            target_matrix = np.array(
                [self.generate_2d_gaussian((input_width, input_height)) * time_point for time_point in
                 freqf_t[:input_depth]])
        else:
            # Use the difference of two Gaussians
            target_matrix = np.array(
                [self.generate_difference_of_2d_gaussians((input_width, input_height), self.surround_weight) * time_point for time_point in
                 freqf_t[:input_depth]])

        return torch.tensor(target_matrix, dtype=torch.float32).to(self.device)

    def generate_difference_of_2d_gaussians(self, size, surround_weight):
        gaussian_matrix1 = self.generate_2d_gaussian(size, self.mean1, self.cov1)
        gaussian_matrix2 = self.generate_2d_gaussian(size, self.mean2, self.cov2)
        return gaussian_matrix1 - surround_weight*gaussian_matrix2

    def generate_2d_gaussian(self, size, mean=None, cov=None):
        if mean is None:
            mean = self.mean1
        if cov is None:
            cov = self.cov1

        x = np.linspace(-1, 1, size[0])
        y = np.linspace(-1, 1, size[1])
        x, y = np.meshgrid(x, y)

        d = np.dstack([x, y])
        gaussian_matrix = np.exp(-0.5 * np.sum(np.dot(d - mean, np.linalg.inv(cov)) * (d - mean), axis=2))

        return gaussian_matrix
