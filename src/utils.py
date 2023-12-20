import torch
import numpy as np

def numpy_to_compatible_tensor(file_name: str, in_channels: int) -> torch.Tensor:
    """
    Convert a numpy array to a tensor compatible with the model.
    """
    input_image = np.load(file_name).astype(np.float32)
    input_tensor = torch.from_numpy(input_image)
    # the input images has no channel dimension, so add it for compatibility
    input_tensor = input_tensor.unsqueeze(0)

    # if in_channels == 3:
    #     # adding two more channels with zeros for compatibility
    #     input_tensor = torch.cat((input_tensor, torch.zeros_like(input_tensor), torch.zeros_like(input_tensor)), dim=0)

    return input_tensor