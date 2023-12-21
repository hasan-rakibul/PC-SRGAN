import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
    
class PhysicsLossInnerImage(nn.Module):
    """Constructs a physics-based loss function for the inner side of the image (w/o boundary pixels).
     """

    def __init__(self, epsilon: float, beta_vec: list, K: float) -> None:
        super().__init__()

        # FEM data generation parameters
        self.epsilon = epsilon
        self.beta_vec = beta_vec # [bx, by]
        self.K = K
        self.delta_t = 0.1 / 100 # T/n_samples

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor, gt_tensor_prev: Tensor) -> Tensor:
        assert sr_tensor.size() == gt_tensor.size() == gt_tensor_prev.size(), "Tensors must have the same size"
        
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

        losses = (
            (-1 / self.delta_t) * (sr_tensor - gt_tensor_prev)
            - self.epsilon * sr_tensor_lap
            + self.beta_vec[0] * sr_tensor_dx + self.beta_vec[1] * sr_tensor_dy
            - self.K * torch.pow(sr_tensor, 2) * (1-sr_tensor)
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

        coeff = - 63.03393257009588 # empirically found through comparing with analytical solutions

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
        # (1, 64, 64) -> (1, 62, 62)
        return tensor[:, 1:-1, 1:-1]
    
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