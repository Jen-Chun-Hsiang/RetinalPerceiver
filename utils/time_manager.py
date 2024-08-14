import time


class TimeFunctionRun:
    def __init__(self, alpha=0.1):
        """
        Initializes the decorator with an alpha parameter which controls the decay rate of the EMA.

        :param alpha: float, the decay factor for the EMA, smaller values give more weight to older data.
        """
        self.alpha = alpha

    def __call__(self, func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)  # Execute the function
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Initialize or update the EMA for this function
            if not hasattr(self, 'timings'):
                self.timings = {}
            if func.__name__ not in self.timings:
                self.timings[func.__name__] = elapsed_time
            else:
                # Update the EMA
                old_ema = self.timings[func.__name__]
                new_ema = self.alpha * elapsed_time + (1 - self.alpha) * old_ema
                self.timings[func.__name__] = new_ema

            return result
        return wrapper
