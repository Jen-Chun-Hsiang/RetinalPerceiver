import math

# Function to convert None values to NaN for MATLAB compatibility
def convert_none_to_nan(timing_dict):
    return {k: (v if v is not None else float('nan')) for k, v in timing_dict.items()}


def convert_nan_to_none(timing_dict):
    return {k: (v if not math.isnan(v) else None) for k, v in timing_dict.items()}