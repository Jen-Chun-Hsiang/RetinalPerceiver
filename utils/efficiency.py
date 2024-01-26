import time
import functools
import numpy as np
from scipy.io import savemat

class Timer:
    def __init__(self):
        self.function_names = []
        self.execution_times = []

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Executed {func.__name__!r} in {elapsed_time:.4f} seconds")

            # Append function name to the list and execution time to another list
            self.function_names.append(func.__name__)
            self.execution_times.append(elapsed_time)

            return result
        return wrapper_timer

    def get_execution_times_as_numpy(self):
        # Convert the execution times list to a NumPy array
        return np.array(self.execution_times)

    def get_function_names(self):
        # Return the list of function names
        return self.function_names

    def save_to_mat(self, filename):
        mat_data = {
            'function_names': np.array(self.function_names, dtype=np.object),
            'execution_times': np.array(self.execution_times)
        }
        savemat(filename, mat_data)