import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
import numpy as np
    
class PhysicsLossInnerImage(nn.Module):
    """Constructs a physics-based loss function for the inner side of the image (w/o boundary pixels).
     """

    def __init__(self) -> None:
        super().__init__()

        # FEM data generation parameters
        self.delta_t = 0.1 / 100 # T/n_samples

    def forward(
            self, sr_tensor: Tensor, gt_tensor: Tensor, gt_tensor_prev: Tensor,
            eps: float, K: float, r: float, theta: float
        ) -> Tensor:
        assert sr_tensor.size() == gt_tensor.size() == gt_tensor_prev.size(), "Tensors must have the same size"
        
        b1 = r * torch.cos(theta) # Velocity component in x direction
        b2 = r * torch.sin(theta) # Velocity component in y direction

        # get device
        self.device = sr_tensor.device

        # input normalization
        # sr_tensor = self.normalize(sr_tensor)
        # gt_tensor = self.normalize(gt_tensor)

        sr_tensor_dx, sr_tensor_dy = self._calculate_image_derivative(sr_tensor)
        sr_tensor_lap = self._calculate_image_laplacian(sr_tensor)

        # remove boundary
        sr_tensor = self._remove_boundary(sr_tensor)
        gt_tensor_prev = self._remove_boundary(gt_tensor_prev)

        # reshape the physics params to (batch, 1, 1, 1) to be able to multiply with tensors
        eps = eps.unsqueeze(1).unsqueeze(2).unsqueeze(3) 
        K = K.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        b1 = b1.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        b2 = b2.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        losses = (
            (1 / self.delta_t) * (sr_tensor - gt_tensor_prev)
            - eps * sr_tensor_lap
            + b1 * sr_tensor_dx + b2 * sr_tensor_dy
            - K * torch.pow(sr_tensor, 2) * (1-sr_tensor)
        )
        
        loss_criteria = nn.MSELoss()
        losses = loss_criteria(losses, torch.zeros_like(losses).to(self.device))

        return losses

    def _calculate_image_derivative(self, img: Tensor) -> [Tensor, Tensor]:        
        # Filter kernels
        k = Tensor([1., 3.5887, 1.]) # or [3., 10., 3.] or [17., 61., 17.]
        d = Tensor([1., 0., -1.])
        gy = torch.outer(k, d)
        gx = gy.transpose(0, 1)

        coeff = - 5.645298778954285 # empirically found through comparing with analytical solutions

        img_dx = coeff * F_torch.conv2d(img, gx.view(1, 1, 3, 3).to(self.device))
        img_dy = coeff * F_torch.conv2d(img, gy.view(1, 1, 3, 3).to(self.device))

        return img_dx, img_dy

    def _calculate_image_laplacian(self, img: Tensor) -> Tensor:

        # Filter kernel
        g = torch.Tensor([[0., 1., 0.],
                        [1., 4., 1.],
                        [0., 1., 0.]])

        coeff = - 9.880939350316519 # empirically found through comparing with analytical solutions
        
        img_lap = coeff * F_torch.conv2d(img, g.view(1, 1, 3, 3).to(self.device))
            
        return img_lap
    
    def _remove_boundary(self, tensor: Tensor) -> Tensor:
        # (batch, 1, 64, 64) -> (batch, 1, 62, 62)
        return tensor[:, :, 1:-1, 1:-1]
    
    def sanity_check(self, img):
        img_dx, img_dy = self._calculate_image_derivative(img)
        img_dx_dx, _ = self._calculate_image_derivative(img_dx)
        _, img_dy_dy = self._calculate_image_derivative(img_dy)

        left = img_dx_dx + img_dy_dy
        print(left, left.shape)
        print('---')

        img_lap = self._calculate_image_laplacian(img)
        img_lap = self._remove_boundary(img_lap)
        # assert left.shape == img_lap.shape
        print(img_lap, img_lap.shape)

        print('Differences:')
        print(left - img_lap)

        print('---')
        print(torch.allclose(img_dx_dx + img_dy_dy, img_lap))

    
class PhysicsLossImageBoundary(nn.Module):
    """Constructs a physics-based loss function for the boundary of the image.
     """

    def __init__(self) -> None:
        super().__init__()

        self.fem_left = 1
        self.fem_right = 0
        self.fem_top = 0
        self.fem_bottom = 0

    def forward(self, sr_tensor: Tensor) -> Tensor:
        
        # get device
        self.device = sr_tensor.device

        # input normalization
        # sr_tensor = self.normalize(sr_tensor)

        left = sr_tensor[:, :, 0]
        right = sr_tensor[:, :, -1]
        top = sr_tensor[:, 0, 1:-1] # due to overalp, remove 1 pixel from each side
        bottom = sr_tensor[:, -1, 1:-1] # due to overalp, remove 1 pixel from each side

        losses_left = F_torch.mse_loss(left, self.fem_left * torch.ones_like(left).to(self.device))
        losses_right = F_torch.mse_loss(right, self.fem_right * torch.ones_like(right).to(self.device))
        losses_top = F_torch.mse_loss(top, self.fem_top * torch.ones_like(top).to(self.device))
        losses_bottom = F_torch.mse_loss(bottom, self.fem_bottom * torch.ones_like(bottom).to(self.device))

        losses = losses_left + losses_right + losses_top + losses_bottom

        return losses