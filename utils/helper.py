import math
import time


# Function to convert None values to NaN for MATLAB compatibility
def convert_none_to_nan(timing_dict):
    return {k: (v if v is not None else float('nan')) if k != 'counter' else v for k, v in timing_dict.items()}


def convert_nan_to_none(timing_dict):
    return {k: (v if not math.isnan(v) else None) for k, v in timing_dict.items()}


class PersistentTimer:
    """
    A persistent context manager for timing code blocks and updating log values with min, max, and moving average.
    """
    def __init__(self, log_values, tau=0.99, n=100):
        self.log_values = log_values
        self.tau = tau
        self.n = n

        # Initialize log values if not already present
        self.log_values.setdefault('min', None)
        self.log_values.setdefault('max', None)
        self.log_values.setdefault('moving_avg', None)
        self.log_values.setdefault('counter', 0)

        # Internal flag to skip timing
        self._timing_needed = False

    def __enter__(self):
        # Increment the counter and check if timing is needed
        self.log_values['counter'] += 1
        if self.log_values['counter'] < self.n:
            self._timing_needed = False
            return None  # Timing is skipped for this iteration
        self._timing_needed = True
        self.start_time = time.perf_counter()  # Start timing
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._timing_needed:
            return  # Skip updating if timing wasn't needed

        end_time = time.perf_counter()
        duration = end_time - self.start_time

        # Handle the first-time setting of values
        if self.log_values['min'] is None:
            self.log_values['min'] = duration
            self.log_values['max'] = duration
            self.log_values['moving_avg'] = duration
        else:
            # Update minimum and maximum
            self.log_values['min'] = min(self.log_values['min'], duration)
            self.log_values['max'] = max(self.log_values['max'], duration)

            # Update moving average with exponential weighting
            self.log_values['moving_avg'] = self.tau * self.log_values['moving_avg'] + (1 - self.tau) * duration

        self.log_values['counter'] = 0  # Reset the counter after the update
