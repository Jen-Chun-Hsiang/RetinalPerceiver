import numpy as np
import torch


class TargetMatrixGenerator:
    def __init__(self, mean=(0, 0), cov=((1, 0), (0, 1)), mean2=None, cov2=None, surround_weight=0.5, device='cpu'):
        self.mean1 = mean
        self.cov1 = cov
        self.mean2 = mean if mean2 is None else mean2
        self.cov2 = cov2
        self.surround_weight = surround_weight
        self.device = device

    def create_3d_target_matrix(self, input_height, input_width, input_depth, tf_weight_surround=0.2,
                                tf_sigma_center=0.05, tf_sigma_surround=0.12, tf_mean_center=0.08,
                                tf_mean_surround=0.12, tf_weight_center=1, tf_offset=0):
        def gaussian(x, sigma, mean_t):
            return np.exp(-np.power(x - mean_t, 2.) / (2 * np.power(sigma, 2.)))

        def Ypf(T, tf_sigma_center, tf_sigma_surround, tf_mean_center, tf_mean_surround,
                tf_weight_center, tf_weight_surround, tf_offset):
            gauss1 = gaussian(T, tf_sigma_center, tf_mean_center) * tf_weight_center
            gauss2 = gaussian(T, tf_sigma_surround, tf_mean_surround) * tf_weight_surround
            return (gauss1 - gauss2) + tf_offset

        SamplingRate = 40
        T = np.arange(-1, 3, 1 / SamplingRate)
        T_positive = T[(T >= 0) & (T < 0.5)]
        freqf_t = Ypf(T_positive, tf_sigma_center, tf_sigma_surround, tf_mean_center, tf_mean_surround,
                      tf_weight_center, tf_weight_surround, tf_offset)

        if self.cov2 is None:
            # Only use the first Gaussian
            target_matrix = np.array(
                [self.generate_2d_gaussian((input_width, input_height)) * time_point for time_point in
                 freqf_t[:input_depth]])
        else:
            # Use the difference of two Gaussians
            target_matrix = np.array(
                [self.generate_difference_of_2d_gaussians((input_width, input_height),
                                                          self.surround_weight) * time_point for time_point in
                 freqf_t[:input_depth]])

        return torch.tensor(target_matrix, dtype=torch.float32).to(self.device)

    def generate_difference_of_2d_gaussians(self, size, surround_weight):
        gaussian_matrix1 = self.generate_2d_gaussian(size, self.mean1, self.cov1)
        gaussian_matrix2 = self.generate_2d_gaussian(size, self.mean2, self.cov2)
        return gaussian_matrix1 - surround_weight * gaussian_matrix2

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


class MultiTargetMatrixGenerator(TargetMatrixGenerator):
    def __init__(self, param_list, device='cpu'):
        super().__init__(device=device)
        self.param_list = param_list

    def create_3d_target_matrices(self, input_height, input_width, input_depth):
        all_matrices = []

        # Ensure self.param_list is always a list of dictionaries
        if isinstance(self.param_list, dict):
            self.param_list = [self.param_list]

        for params in self.param_list:
            self.mean1, self.cov1, self.mean2, self.cov2, self.surround_weight = (
                params['sf_mean_center'], params['sf_cov_center'], params['sf_mean_surround'],
                params['sf_cov_surround'], params['sf_weight_surround']
            )
            matrix = self.create_3d_target_matrix(
                input_height,
                input_width,
                input_depth,
                params['tf_weight_surround'],
                params['tf_sigma_center'],
                params['tf_sigma_surround'],
                params['tf_mean_center'],
                params['tf_mean_surround'],
                params['tf_weight_center'],
                params['tf_offset']
            )
            all_matrices.append(matrix.unsqueeze(0))  # Add an extra dimension for concatenation

        return torch.cat(all_matrices, dim=0)  # Concatenating along the new dimension


def create_hexagonal_centers(xlim, ylim, target_num_centers, max_iterations=100, noise_level=0.3, set_rand_seed=None):
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Set the random seed if specified
    if set_rand_seed is not None:
        np.random.seed(set_rand_seed)

    # Estimate initial side length based on target number of centers
    approximate_area = x_range * y_range
    approximate_cell_area = approximate_area / target_num_centers
    side_length = np.sqrt(approximate_cell_area / (3 * np.sqrt(3) / 2))

    # Calculate horizontal and vertical spacing
    dx = side_length * 3 ** 0.5
    dy = side_length * 1.5

    # Estimate number of columns and rows
    cols = int(np.ceil(x_range / dx))
    rows = int(np.ceil(y_range / dy))

    # Function to generate grid points with given spacing and offsets
    def generate_points_with_noise(dx, dy, offset_x, offset_y, noise_level):
        points = []
        for row in range(rows):
            for col in range(cols):
                x = col * dx + x_min - offset_x
                y = row * dy + y_min - offset_y
                if row % 2 == 1:
                    x += dx / 2  # Offset every other row

                # Add noise
                x += np.random.uniform(-noise_level, noise_level) * dx
                y += np.random.uniform(-noise_level, noise_level) * dy

                if x_min <= x < x_max and y_min <= y < y_max:
                    points.append((x, y))
        return points

    # Calculate initial offsets to center the grid
    offset_x = (cols * dx - x_range) / 2
    offset_y = (rows * dy - y_range) / 2

    # Generate initial grid points
    points = generate_points_with_noise(dx, dy, offset_x, offset_y, noise_level)

    # Adjust grid spacing for a fixed number of iterations
    for _ in range(max_iterations):
        if len(points) > target_num_centers:
            dx *= 1.01
            dy *= 1.01
        else:
            dx *= 0.99
            dy *= 0.99
        cols = int(np.ceil(x_range / dx))
        rows = int(np.ceil(y_range / dy))
        offset_x = (cols * dx - x_range) / 2
        offset_y = (rows * dy - y_range) / 2
        points = generate_points_with_noise(dx, dy, offset_x, offset_y, noise_level)
        if abs(len(points) - target_num_centers) <= target_num_centers * 0.05:  # 5% tolerance
            break

    return points


class CellLevel:
    def __init__(self, sf_mean_center=None, sf_mean_surround=None):
        self.sf_mean_center = sf_mean_center
        self.sf_mean_surround = sf_mean_surround

    def get_params(self):
        return {
            'sf_mean_center': self.sf_mean_center,
            'sf_mean_surround': self.sf_mean_surround
        }


class CellClassLevel:
    def __init__(self, sf_cov_center, sf_cov_surround, sf_weight_surround, num_cells, xlim, ylim, class_level_id,
                 set_rand_seed):
        self.sf_cov_center = sf_cov_center
        self.sf_cov_surround = sf_cov_surround
        self.sf_weight_surround = sf_weight_surround
        self.class_level_id = class_level_id  # New: Unique identifier for the cell class level
        self.cells = self.create_cells(num_cells, xlim, ylim, set_rand_seed)

    def create_cells(self, num_cells, xlim, ylim, set_rand_seed):
        centers = create_hexagonal_centers(xlim, ylim, num_cells, set_rand_seed=set_rand_seed)
        cells = []
        for center in centers:
            cell = CellLevel(sf_mean_center=center, sf_mean_surround=center)
            cells.append(cell)
        return cells

    def get_params(self):
        params = []
        for cell in self.cells:
            cell_params = cell.get_params()
            cell_params.update({
                'sf_cov_center': self.sf_cov_center,
                'sf_cov_surround': self.sf_cov_surround,
                'sf_weight_surround': self.sf_weight_surround,
                'class_level_id': self.class_level_id  # Include the unique class level ID in each cell's parameters
            })
            params.append(cell_params)
        return params


class ExperimentalLevel:
    def __init__(self, tf_weight_surround, tf_sigma_center, tf_sigma_surround, tf_mean_center, tf_mean_surround,
                 tf_weight_center, tf_offset, cell_classes):
        """
        Initialize the experimental level with specific parameters and a list of cell classes.

        :param tf_weight_surround: Parameter for the weight surround at the experimental level.
        :param tf_sigma_center: Sigma center parameter at the experimental level.
        :param tf_sigma_surround: Sigma surround parameter at the experimental level.
        :param tf_mean_center: Mean center parameter at the experimental level.
        :param tf_mean_surround: Mean surround parameter at the experimental level.
        :param tf_weight_center: Weight center parameter at the experimental level.
        :param tf_offset: Offset parameter at the experimental level.
        :param cell_classes: A list of CellClassLevel objects.
        """
        self.tf_weight_surround = tf_weight_surround
        self.tf_sigma_center = tf_sigma_center
        self.tf_sigma_surround = tf_sigma_surround
        self.tf_mean_center = tf_mean_center
        self.tf_mean_surround = tf_mean_surround
        self.tf_weight_center = tf_weight_center
        self.tf_offset = tf_offset
        self.cell_classes = cell_classes

    def generate_param_list(self, global_cell_id_start):
        """
        Generates a parameter list for this experimental level, including cell class level information.
        Does not descend to the individual cell level.

        :return: A list of tuples, each containing cell class parameters and a list of cell parameters.
        """
        param_list = []
        global_cell_id = global_cell_id_start

        for cell_class in self.cell_classes:
            class_params = cell_class.get_params()

            for cell_params in class_params:
                cell_params_with_global_id = {**cell_params, 'cell_level_id': global_cell_id}
                global_cell_id += 1  # Increment for the next cell
                param_list.append({
                    'tf_weight_surround': self.tf_weight_surround,
                    'tf_sigma_center': self.tf_sigma_center,
                    'tf_sigma_surround': self.tf_sigma_surround,
                    'tf_mean_center': self.tf_mean_center,
                    'tf_mean_surround': self.tf_mean_surround,
                    'tf_weight_center': self.tf_weight_center,
                    'tf_offset': self.tf_offset,
                    'class_level_id': cell_class.class_level_id,  # Directly use class_level_id from cell_class
                    'cell_params': [cell_params_with_global_id]
                })

        return param_list, global_cell_id


class IntegratedLevel:
    def __init__(self, experimental_levels, is_coordinates=False):
        """
        Initialize the integrated level with a list of ExperimentalLevel instances.

        :param experimental_levels: A list of ExperimentalLevel objects.
        """
        self.experimental_levels = experimental_levels
        self.is_coordinates = is_coordinates

    def generate_combined_param_list(self):
        """
        Generates a comprehensive parameter list and unique series IDs across all levels.

        :return: A tuple containing two lists:
                 1. A combined list of dictionaries, each containing a set of parameters.
                 2. A list of unique series IDs corresponding to each entry in the parameter list.
        """
        combined_param_list = []
        series_ids = []
        global_cell_id_start = 1  # Start ID for global cell uniqueness

        for exp_level_id, exp_level in enumerate(self.experimental_levels, start=1):
            class_param_list, global_cell_id_start = exp_level.generate_param_list(global_cell_id_start)

            for class_params in class_param_list:
                # Here, class_params already includes 'class_level_id' and 'cell_params' with 'cell_level_id'
                cell_params = class_params['cell_params'][0]  # Assuming 'cell_params' list is not empty
                combined_params = {
                    **{key: class_params[key] for key in ['tf_weight_surround', 'tf_sigma_center', 'tf_sigma_surround',
                                                          'tf_mean_center', 'tf_mean_surround', 'tf_weight_center',
                                                          'tf_offset']},
                    **{key: cell_params[key] for key in ['sf_cov_center', 'sf_cov_surround',
                                                         'sf_weight_surround', 'sf_mean_center', 'sf_mean_surround']}
                }
                combined_param_list.append(combined_params)
                if self.is_coordinates:
                    series_ids.append((exp_level_id, class_params['class_level_id'],
                                       combined_params['sf_mean_center'][0], combined_params['sf_mean_center'][1]))
                else:
                    series_ids.append((exp_level_id, class_params['class_level_id'], cell_params['cell_level_id']))

        return combined_param_list, series_ids
