import numpy as np
import torch
import random


class TargetMatrixGenerator:
    def __init__(self, mean=(0, 0), cov=((1, 0), (0, 1)), mean2=None, cov2=None, surround_weight=0.5,
                 is_norm_matrix=False):
        self.mean1 = mean
        self.cov1 = cov
        self.mean2 = mean if mean2 is None else mean2
        self.cov2 = cov2
        self.surround_weight = surround_weight
        self.is_norm_matrix = is_norm_matrix

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
            gaussian = self.generate_2d_gaussian((input_width, input_height))
        else:
            gaussian = self.generate_difference_of_2d_gaussians((input_width, input_height), self.surround_weight)

        if self.is_norm_matrix:
            gaussian = gaussian - np.median(gaussian)
            gaussian = gaussian / np.sum(np.abs(gaussian))
            # normalize temporal filter
            freqf_t = freqf_t / np.sum(np.abs(freqf_t))

        # Convert to PyTorch tensors
        gaussian = torch.tensor(gaussian.copy(), dtype=torch.float32).unsqueeze(-1)  # Shape [w, h]
        freqf_t = torch.tensor(freqf_t.copy(), dtype=torch.float32).T  # Shape [t, 1]

        # Broadcasting happens automatically for element-wise multiplication
        target_matrix = gaussian * freqf_t  # Shape [w, h, t]

        return target_matrix

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
    def __init__(self, param_list, is_norm_matrix=False):
        super().__init__(is_norm_matrix=is_norm_matrix)
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


def generate_sf_parameters(row_id, sf_param_table):
    row = sf_param_table.iloc[row_id]
    sf_cov_center = np.array([[row['sf_cov_center_1'], row['sf_cov_center_2']],
                              [row['sf_cov_center_3'], row['sf_cov_center_4']]])
    sf_cov_surround = np.array([[row['sf_cov_surround_1'], row['sf_cov_surround_2']],
                                [row['sf_cov_surround_3'], row['sf_cov_surround_4']]])
    sf_weight_surround = row['sf_weight_surround']
    max_sf_stretch_fac = row['max_sf_stretch_fac']
    return sf_cov_center, sf_cov_surround, sf_weight_surround, max_sf_stretch_fac


def generate_tf_parameters(row_id, tf_param_table):
    row = tf_param_table.iloc[row_id]
    tf_weight_center = row['tf_weight_center']
    tf_weight_surround = row['tf_weight_surround']
    tf_mean_center = row['tf_mean_center']
    tf_mean_surround = row['tf_mean_surround']
    tf_sigma_center = row['tf_sigma_center']
    tf_sigma_surround = row['tf_sigma_surround']
    max_tf_stretch_fac = row['max_tf_stretch_fac']
    return tf_weight_center, tf_weight_surround, tf_mean_center, tf_mean_surround, tf_sigma_center, tf_sigma_surround, max_tf_stretch_fac


def generate_coordinates(posi_id, posi_cache):
    if posi_id not in posi_cache:
        sf_mean_center = (random.uniform(-1, 1), random.uniform(-1, 1))
        posi_cache[posi_id] = sf_mean_center
    else:
        sf_mean_center = posi_cache[posi_id]

    sf_mean_surround = sf_mean_center
    return sf_mean_center, sf_mean_surround


def apply_batch_scaling(batch_value, tf_mean_center, tf_mean_surround, tf_sigma_center, tf_sigma_surround,
                        max_stretch_fac):
    scale_factor = 1 + (batch_value * (max_stretch_fac - 1))
    tf_mean_center *= scale_factor
    tf_mean_surround *= scale_factor
    tf_sigma_center *= scale_factor
    tf_sigma_surround *= scale_factor
    return tf_mean_center, tf_mean_surround, tf_sigma_center, tf_sigma_surround


def apply_eccentricity_scaling(eccentricity, sf_cov_center, sf_cov_surround, max_stretch_fac):
    scale_factor = 1 + (eccentricity * (max_stretch_fac - 1))
    sf_cov_center *= scale_factor
    sf_cov_surround *= scale_factor
    return sf_cov_center, sf_cov_surround


def assign_row_ids_to_cell_ids(unique_cell_ids, num_sf_rows, num_tf_rows, random_state):
    if len(unique_cell_ids) > num_sf_rows or len(unique_cell_ids) > num_tf_rows:
        raise ValueError("Number of unique cell IDs exceeds available row IDs.")

    # Shuffle unique_cell_ids to ensure random assignment
    random_state.shuffle(unique_cell_ids)

    sf_row_ids = random_state.sample(range(num_sf_rows), len(unique_cell_ids))
    tf_row_ids = random_state.sample(range(num_tf_rows), len(unique_cell_ids))

    cell_id_to_row_ids = {cell_id: (sf_row_ids[i], tf_row_ids[i]) for i, cell_id in enumerate(unique_cell_ids)}
    return cell_id_to_row_ids


import random

class ParameterGenerator:
    def __init__(self, sf_param_table, tf_param_table, seed=None):
        self.sf_param_table = sf_param_table
        self.tf_param_table = tf_param_table

        # Initialize caches for batch values and eccentricities
        self.batch_cache = {}
        self.eccentricity_cache = {}

        # Use a unique seed per instance to ensure different randomness across instances
        if seed is None:
            self.instance_seed = random.randint(0, int(1e9))
        else:
            self.instance_seed = seed
        self.random_state = random.Random(self.instance_seed)

        # Initialize other properties to None
        self.unique_posi_ids = None
        self.posi_id_to_center = None
        self.unique_cell_ids = None
        self.cell_id_to_row_ids = None
        self.min_max_stretch_fac = None
        self.eccentricity_exists = None

    def generate_parameters(self, query_df):
        param_list = []
        query_list = []
        self.unique_posi_ids = list(query_df['Posi_id'].unique())
        hex_centers = create_hexagonal_centers(
            xlim=(0, 1),
            ylim=(0, 1),
            target_num_centers=len(self.unique_posi_ids),
            set_rand_seed=self.instance_seed  # Use instance-specific seed
        )
        self.posi_id_to_center = {posi_id: hex_centers[i] for i, posi_id in enumerate(self.unique_posi_ids)}

        self.unique_cell_ids = list(query_df['Cell_id'].unique())
        self.cell_id_to_row_ids = assign_row_ids_to_cell_ids(
            self.unique_cell_ids, len(self.sf_param_table), len(self.tf_param_table), self.random_state
        )

        # Determine min value of 'max_tf_stretch_fac' for the selected tf_row_ids
        self.min_max_stretch_fac = self.tf_param_table.loc[
            [self.cell_id_to_row_ids[cell_id][1] for cell_id in self.unique_cell_ids],
            'max_tf_stretch_fac'
        ].min()

        # Check if the "Eccentricity" column exists
        self.eccentricity_exists = 'Eccentricity' in query_df.columns

        # Precompute and store batch values
        unique_batch_ids = query_df['Batch_id'].unique()
        for batch_id in unique_batch_ids:
            if batch_id not in self.batch_cache:
                self.batch_cache[batch_id] = self.random_state.uniform(0, 1)

        # Precompute and store eccentricity values if needed
        if self.eccentricity_exists:
            batch_ids_with_ecc_minus_one = query_df.loc[query_df['Eccentricity'] == -1, 'Batch_id'].unique()
            for batch_id in batch_ids_with_ecc_minus_one:
                if batch_id not in self.eccentricity_cache:
                    self.eccentricity_cache[batch_id] = self.random_state.uniform(0, 1)

        for _, row in query_df.iterrows():
            batch_id = int(row['Batch_id'])
            cell_id = int(row['Cell_id'])
            posi_id = int(row['Posi_id'])

            if self.eccentricity_exists:
                eccentricity_code = float(row['Eccentricity'])

            # Retrieve batch value from cache
            batch_value = self.batch_cache[batch_id]

            # Retrieve eccentricity from cache if needed
            if self.eccentricity_exists:
                if eccentricity_code == -1:
                    eccentricity = self.eccentricity_cache[batch_id]
                else:
                    eccentricity = eccentricity_code

            sf_row_id, tf_row_id = self.cell_id_to_row_ids[cell_id]

            params = {
                'tf_weight_surround': None,
                'tf_sigma_center': None,
                'tf_sigma_surround': None,
                'tf_mean_center': None,
                'tf_mean_surround': None,
                'tf_weight_center': None,
                'tf_offset': 0,
                'sf_cov_center': None,
                'sf_cov_surround': None,
                'sf_weight_surround': None,
                'sf_mean_center': None,
                'sf_mean_surround': None
            }

            # Generate temporal and spatial frequency parameters
            (params['tf_weight_center'], params['tf_weight_surround'],
             tf_mean_center, tf_mean_surround,
             tf_sigma_center, tf_sigma_surround, local_max_tf_stretch_fac) = generate_tf_parameters(
                tf_row_id, self.tf_param_table)
            (sf_cov_center, sf_cov_surround, params['sf_weight_surround'],
             local_max_sf_stretch_fac) = generate_sf_parameters(
                sf_row_id, self.sf_param_table)

            # Assign position-based centers
            params['sf_mean_center'] = self.posi_id_to_center[posi_id]

            # Apply batch scaling for temporal frequency parameters
            (tf_mean_center, tf_mean_surround,
             tf_sigma_center, tf_sigma_surround) = apply_batch_scaling(
                batch_value, tf_mean_center, tf_mean_surround,
                tf_sigma_center, tf_sigma_surround, self.min_max_stretch_fac)

            params['tf_mean_center'] = tf_mean_center
            params['tf_mean_surround'] = tf_mean_surround
            params['tf_sigma_center'] = tf_sigma_center
            params['tf_sigma_surround'] = tf_sigma_surround

            # If eccentricity exists, apply eccentricity scaling to spatial frequency parameters
            if self.eccentricity_exists:
                (sf_cov_center, sf_cov_surround) = apply_eccentricity_scaling(
                    eccentricity, sf_cov_center, sf_cov_surround, local_max_sf_stretch_fac)

            params['sf_cov_center'] = sf_cov_center
            params['sf_cov_surround'] = sf_cov_surround

            # Append to result lists
            param_list.append(params)
            if self.eccentricity_exists:
                query_list.append((batch_id, cell_id, eccentricity_code,
                                   params['sf_mean_center'][0], params['sf_mean_center'][1]))
            else:
                query_list.append((batch_id, cell_id,
                                   params['sf_mean_center'][0], params['sf_mean_center'][1]))

        return param_list, query_list

    def generate_parameters_from_query_list(self, query_list):
        param_list = []

        if not query_list:
            return param_list  # Return an empty list if query_list is empty

        # Ensure that required attributes are available
        required_attributes = ['cell_id_to_row_ids', 'min_max_stretch_fac',
                               'eccentricity_exists', 'batch_cache', 'eccentricity_cache']
        for attr in required_attributes:
            if getattr(self, attr) is None:
                raise ValueError(f"Attribute '{attr}' is not set. Please run 'generate_parameters' first.")

        # Use the stored attributes
        cell_id_to_row_ids = self.cell_id_to_row_ids
        min_max_stretch_fac = self.min_max_stretch_fac
        eccentricity_exists = self.eccentricity_exists

        for item in query_list:
            if eccentricity_exists:
                batch_id, cell_id, eccentricity_code, sf_mean_center_x, sf_mean_center_y = item
            else:
                batch_id, cell_id, sf_mean_center_x, sf_mean_center_y = item

            # Retrieve batch value from cache
            if batch_id in self.batch_cache:
                batch_value = self.batch_cache[batch_id]
            else:
                raise ValueError(f"Batch ID {batch_id} not found in batch cache.")

            # Retrieve eccentricity from cache if needed
            if eccentricity_exists:
                if eccentricity_code == -1:
                    if batch_id in self.eccentricity_cache:
                        eccentricity = self.eccentricity_cache[batch_id]
                    else:
                        # Choose a random value from self.eccentricity_cache
                        if self.eccentricity_cache:
                            eccentricity = self.random_state.choice(list(self.eccentricity_cache.values()))
                        else:
                            # If the cache is empty, generate a new random value and store it
                            eccentricity = self.random_state.uniform(0, 1)
                            self.eccentricity_cache[batch_id] = eccentricity
                else:
                    eccentricity = eccentricity_code

            sf_row_id, tf_row_id = cell_id_to_row_ids[cell_id]

            params = {
                'tf_weight_surround': None,
                'tf_sigma_center': None,
                'tf_sigma_surround': None,
                'tf_mean_center': None,
                'tf_mean_surround': None,
                'tf_weight_center': None,
                'tf_offset': 0,
                'sf_cov_center': None,
                'sf_cov_surround': None,
                'sf_weight_surround': None,
                'sf_mean_center': None,
                'sf_mean_surround': None
            }

            # Generate temporal and spatial frequency parameters
            (params['tf_weight_center'], params['tf_weight_surround'],
             tf_mean_center, tf_mean_surround,
             tf_sigma_center, tf_sigma_surround, local_max_tf_stretch_fac) = generate_tf_parameters(
                tf_row_id, self.tf_param_table)
            (sf_cov_center, sf_cov_surround, params['sf_weight_surround'],
             local_max_sf_stretch_fac) = generate_sf_parameters(
                sf_row_id, self.sf_param_table)

            # Assign sf_mean_center from input
            params['sf_mean_center'] = (sf_mean_center_x, sf_mean_center_y)

            # Apply batch scaling for temporal frequency parameters
            (tf_mean_center, tf_mean_surround,
             tf_sigma_center, tf_sigma_surround) = apply_batch_scaling(
                batch_value, tf_mean_center, tf_mean_surround,
                tf_sigma_center, tf_sigma_surround, min_max_stretch_fac)

            params['tf_mean_center'] = tf_mean_center
            params['tf_mean_surround'] = tf_mean_surround
            params['tf_sigma_center'] = tf_sigma_center
            params['tf_sigma_surround'] = tf_sigma_surround

            # If eccentricity exists, apply eccentricity scaling to spatial frequency parameters
            if eccentricity_exists:
                (sf_cov_center, sf_cov_surround) = apply_eccentricity_scaling(
                    eccentricity, sf_cov_center, sf_cov_surround, local_max_sf_stretch_fac)

            params['sf_cov_center'] = sf_cov_center
            params['sf_cov_surround'] = sf_cov_surround

            # Append to result list
            param_list.append(params)

        return param_list


def generate_parameters(query_df, sf_param_table, tf_param_table):
    param_list = []
    query_list = []
    batch_cache = {}
    posi_cache = {}
    eccentricity_cache = {}
    set_rand_seed = 42

    unique_posi_ids = list(query_df['Posi_id'].unique())
    hex_centers = create_hexagonal_centers(xlim=(0, 1), ylim=(0, 1), target_num_centers=len(unique_posi_ids),
                                           set_rand_seed=set_rand_seed)
    posi_id_to_center = {posi_id: hex_centers[i] for i, posi_id in enumerate(unique_posi_ids)}

    unique_cell_ids = list(query_df['Cell_id'].unique())
    cell_id_to_row_ids = assign_row_ids_to_cell_ids(unique_cell_ids, len(sf_param_table), len(tf_param_table))

    # Determine min value of 'max_tf_stretch_fac' for the selected tf_row_ids
    min_max_stretch_fac = tf_param_table.loc[
        [cell_id_to_row_ids[cell_id][1] for cell_id in unique_cell_ids], 'max_tf_stretch_fac'].min()

    # Check if the "Eccentricity" column exists
    eccentricity_exists = 'Eccentricity' in query_df.columns

    for _, row in query_df.iterrows():
        batch_id = int(row['Batch_id'])
        cell_id = int(row['Cell_id'])
        posi_id = int(row['Posi_id'])

        if eccentricity_exists:
            eccentricity = float(row['Eccentricity'])
            eccentricity_code = eccentricity

        # Handle batch value caching
        if batch_id not in batch_cache:
            batch_cache[batch_id] = random.uniform(0, 1)
        batch_value = batch_cache[batch_id]

        # If eccentricity exists and is -1, assign a batch-based random value
        if eccentricity_exists and eccentricity_code == -1:
            if batch_id not in eccentricity_cache:
                eccentricity_cache[batch_id] = random.uniform(0, 1)
            eccentricity = eccentricity_cache[batch_id]

        sf_row_id, tf_row_id = cell_id_to_row_ids[cell_id]

        params = {
            'tf_weight_surround': None,
            'tf_sigma_center': None,
            'tf_sigma_surround': None,
            'tf_mean_center': None,
            'tf_mean_surround': None,
            'tf_weight_center': None,
            'tf_offset': 0,
            'sf_cov_center': None,
            'sf_cov_surround': None,
            'sf_weight_surround': None,
            'sf_mean_center': None,
            'sf_mean_surround': None
        }

        # Generate temporal and spatial frequency parameters
        params['tf_weight_center'], params[
            'tf_weight_surround'], tf_mean_center, tf_mean_surround, tf_sigma_center, tf_sigma_surround, max_tf_stretch_fac = generate_tf_parameters(
            tf_row_id, tf_param_table)
        sf_cov_center, sf_cov_surround, params['sf_weight_surround'], max_sf_stretch_fac = generate_sf_parameters(
            sf_row_id, sf_param_table)

        # Assign position-based centers
        params['sf_mean_center'] = posi_id_to_center[posi_id]
        params['sf_mean_center'] = params['sf_mean_center']

        # Apply batch scaling for temporal frequency parameters
        tf_mean_center, tf_mean_surround, tf_sigma_center, tf_sigma_surround = apply_batch_scaling(
            batch_value, tf_mean_center, tf_mean_surround, tf_sigma_center, tf_sigma_surround, min_max_stretch_fac)

        params['tf_mean_center'] = tf_mean_center
        params['tf_mean_surround'] = tf_mean_surround
        params['tf_sigma_center'] = tf_sigma_center
        params['tf_sigma_surround'] = tf_sigma_surround

        # If eccentricity exists, apply eccentricity scaling to spatial frequency parameters
        if eccentricity_exists:
            sf_cov_center, sf_cov_surround = apply_eccentricity_scaling(eccentricity, sf_cov_center, sf_cov_surround,
                                                                        max_sf_stretch_fac)

        params['sf_cov_center'] = sf_cov_center
        params['sf_cov_surround'] = sf_cov_surround

        # Append to result lists
        param_list.append(params)
        if eccentricity_exists:
            query_list.append((batch_id, cell_id, eccentricity_code, params['sf_mean_center'][0], params['sf_mean_center'][1]))
        else:
            query_list.append((batch_id, cell_id, params['sf_mean_center'][0], params['sf_mean_center'][1]))

    return param_list, query_list

