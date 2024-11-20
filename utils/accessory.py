import torch

def calculate_mask_positions(lengths, masking_pos):
    # Calculate start positions for each field based on cumulative sums of lengths
    fields = list(lengths.keys())
    positions = {field: sum(lengths[fields[i]] for i in range(index))
                 for index, field in enumerate(fields)}

    # Prepare a list to collect all indices that need to be masked
    mask_indices = []

    # Add ranges to the mask_indices list based on masking_pos
    for pos in masking_pos:
        field = fields[pos]
        start = positions[field]
        end = start + lengths[field]
        mask_indices.extend(range(start, end))

    # Create a tensor from the list of mask_indices
    return torch.tensor(mask_indices)