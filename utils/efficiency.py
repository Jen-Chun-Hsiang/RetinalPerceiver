import time
import functools
import numpy as np

class Timer:
    def __init__(self):
        self.execution_data = []

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Executed {func.__name__!r} in {elapsed_time:.4f} seconds")

            # Append function name and execution time to the list
            self.execution_data.append((func.__name__, elapsed_time))

            return result
        return wrapper_timer

    def get_execution_times_as_numpy(self):
        return np.array(self.execution_data, dtype=[('function', 'U20'), ('time', 'f4')])
