import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# require LaTeX for scienceplots. Comment the following line if you have no LaTeX installed
# may become slower by the way
plt.style.use(['science', 'tableau-colorblind10'])

def numpy_to_compatible_tensor(file_name: str, in_channels: int) -> torch.Tensor:
    """
    Convert a numpy array to a tensor compatible with the model.
    """
    input_image = np.load(file_name).astype(np.float32)
    input_tensor = torch.from_numpy(input_image)
    # the input images has no channel dimension, so add it for compatibility
    input_tensor = input_tensor.unsqueeze(0)

    # scaling the input image to [-1, 1]
    # input_tensor = input_tensor / torch.max(torch.abs(input_tensor))

    # rescaling from [-1, 1] to [0, 1]
    # input_tensor = (input_tensor + 1) / 2

    # if in_channels == 3:
    #     # adding two more channels with zeros for compatibility
    #     input_tensor = torch.cat((input_tensor, torch.zeros_like(input_tensor), torch.zeros_like(input_tensor)), dim=0)

    return input_tensor

def save_as_plot(input_data: np.ndarray, file_name: str) -> None:
    """
    Save a numpy array as matplotlib plot.
    input_data shape: (height, width)
    """
    plt.imshow(input_data, cmap='jet', extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()