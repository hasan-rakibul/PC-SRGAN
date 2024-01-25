import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
    
class PhysicsLossInnerImage(nn.Module):
    """Constructs a physics-based loss function for the inner side of the image (w/o boundary pixels).
     """

    def __init__(self) -> None:
        super().__init__()

        # FEM data generation parameters
        self.delta_t = 0.1 / 100 # T/n_samples

    def forward(
            self, sr_tensor: Tensor, gt_tensor: Tensor, gt_tensor_prev: Tensor, gt_tensor_two_prev: Tensor,
            eps: Tensor, K: Tensor, r: Tensor, theta: Tensor
        ) -> Tensor:
        assert sr_tensor.size() == gt_tensor.size() == gt_tensor_prev.size(), "Tensors must have the same size"

        # get device
        self.device = sr_tensor.device

        b1 = r * torch.cos(theta) # Velocity component in x direction
        b2 = r * torch.sin(theta) # Velocity component in y direction

        # reshape the physics params to (batch, 1, 1, 1) to be able to multiply with tensors
        eps = eps.unsqueeze(1).unsqueeze(2).unsqueeze(3) 
        K = K.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        b1 = b1.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        b2 = b2.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # remove boundary
        sr_tensor_wo_bd = self._remove_boundary(sr_tensor)
        gt_tensor_prev_wo_bd = self._remove_boundary(gt_tensor_prev)
        gt_tensor_two_prev_wo_bd = self._remove_boundary(gt_tensor_two_prev)

        # losses = (
        #     (1 / self.delta_t) * (sr_tensor_wo_bd - gt_tensor_prev_wo_bd)
        #     + (1/2) * (self._calculate_spatial_operators(eps, K, b1, b2, sr_tensor) + self._calculate_spatial_operators(eps, K, b1, b2, gt_tensor_prev))
        # )
        # losses = (
        #     (1 / self.delta_t) * (sr_tensor_wo_bd - gt_tensor_prev_wo_bd)
        #     + self._calculate_spatial_operators(eps, K, b1, b2, gt_tensor_prev)
        # )
        losses = (
            3/2 * (sr_tensor_wo_bd - 4/3*gt_tensor_prev_wo_bd+ 1/3*gt_tensor_two_prev_wo_bd)
            + self.delta_t* self._calculate_spatial_operators(eps, K, b1, b2, sr_tensor)
        )
        # print('first part: ', (sr_tensor_wo_bd - 4/3*gt_tensor_prev_wo_bd+ 1/3*gt_tensor_two_prev_wo_bd))
        # print('spatial op: ', self._calculate_spatial_operators(eps, K, b1, b2, sr_tensor))

        
        loss_criteria = nn.MSELoss()
        losses = loss_criteria(losses, torch.zeros_like(losses).to(self.device))

        # print('loss: ', losses)
        return losses

    def _calculate_spatial_operators(self, eps: Tensor, K: Tensor, b1:Tensor, b2:Tensor, img: Tensor) -> Tensor:
        # input normalization
        # sr_tensor = self.normalize(sr_tensor)
        # gt_tensor = self.normalize(gt_tensor)

        img_dx, img_dy = self._calculate_image_derivative(img)
        img_lap = self._calculate_image_laplacian(img)

        img = self._remove_boundary(img)

        spatial_op = (
            - eps * img_lap
            + b1 * img_dx + b2 * img_dy
            - K * torch.pow(img, 2) * (1-img)
        )
        
        return spatial_op

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
        # (batch, channel, 64, 64) -> (batch, channel, 62, 62)
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
        print(img_lap, img_lap.shape)
        assert left.shape == img_lap.shape

        print('Differences:')
        print(left - img_lap)

        print('---')
        print(torch.allclose(img_dx_dx + img_dy_dy, img_lap))

    
class PhysicsLossImageBoundary(nn.Module):
    """Constructs a physics-based loss function for the boundary of the image.
     """

    def __init__(self) -> None:
        super().__init__()

        # self.fem_left = 1
        # self.fem_right = 0
        # self.fem_top = 0
        # self.fem_bottom = 0

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        
        # get device
        self.device = sr_tensor.device

        # input normalization
        # sr_tensor = self.normalize(sr_tensor)

        # sr_tensor shape: (batch, channel, 64, 64)
        # left = sr_tensor[:, :, 1:-1, 0] # due to overalp, remove 1 pixel from top and bottom
        # right = sr_tensor[:, :, :, -1]
        # top = sr_tensor[:, :, 0, :] 
        # bottom = sr_tensor[:, :, -1, :]

        # losses_left = F_torch.mse_loss(left, self.fem_left * torch.ones_like(left).to(self.device))
        # losses_right = F_torch.mse_loss(right, self.fem_right * torch.ones_like(right).to(self.device))
        # losses_top = F_torch.mse_loss(top, self.fem_top * torch.ones_like(top).to(self.device))
        # losses_bottom = F_torch.mse_loss(bottom, self.fem_bottom * torch.ones_like(bottom).to(self.device))

        # losses_top = F_torch.mse_loss(sr_tensor[:, :, 0, :], gt_tensor[:, :, 0, :]).to(self.device)
        # losses_bottom = F_torch.mse_loss(sr_tensor[:, :, -1, :], gt_tensor[:, :, -1, :]).to(self.device)
        # losses_left = F_torch.mse_loss(sr_tensor[:, :, 1:-1, 0], gt_tensor[:, :, 1:-1, 0]).to(self.device) # due to overalp, remove 1 pixel from top and bottom
        # losses_right = F_torch.mse_loss(sr_tensor[:, :, 1:-1, -1], gt_tensor[:, :, 1:-1, -1]).to(self.device)

        losses_top_bottom = F_torch.mse_loss(sr_tensor[:, :, 0, :], sr_tensor[:, :, -1, :]).to(self.device)
        losses_left_right = F_torch.mse_loss(sr_tensor[:, :, 1:-1, 0], sr_tensor[:, :, 1:-1, -1]).to(self.device) # due to overalp, remove 1 pixel from top and bottom

        losses = losses_left_right + losses_top_bottom

        # losses = losses_top + losses_bottom + losses_left + losses_right

        return losses
    
class PhysicsLossInnerImageAllenCahn(PhysicsLossInnerImage):
    """Constructs a physics-based loss function for the inner side of the image (w/o boundary pixels).
     """

    def __init__(self) -> None:
        super().__init__()

        # FEM data generation parameters
        self.delta_t = 0.001 / 100 # T/n_samples

    def forward(
            self, sr_tensor: Tensor, gt_tensor: Tensor, gt_tensor_prev: Tensor, gt_tensor_two_prev: Tensor,
            eps: Tensor, K: Tensor, r: Tensor, theta: Tensor
        ) -> Tensor:
        assert sr_tensor.size() == gt_tensor.size() == gt_tensor_prev.size(), "Tensors must have the same size"

        # get device
        self.device = sr_tensor.device

        b1 = r * torch.cos(theta) # Velocity component in x direction
        b2 = r * torch.sin(theta) # Velocity component in y direction

        # reshape the physics params to (batch, 1, 1, 1) to be able to multiply with tensors
        eps = eps.unsqueeze(1).unsqueeze(2).unsqueeze(3) 
        K = K.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        b1 = b1.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        b2 = b2.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        theta = theta.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # normalisation by dividing by max(gt_tensor)
        # divisor = 1.1*torch.max(torch.abs(gt_tensor))
        # sr_tensor = sr_tensor / divisor
        # gt_tensor = gt_tensor / divisor
        # gt_tensor_prev = gt_tensor_prev / divisor
        # gt_tensor_two_prev = gt_tensor_two_prev / divisor

        # remove boundary
        sr_tensor_wo_bd = self._remove_boundary(sr_tensor)
        gt_tensor_prev_wo_bd = self._remove_boundary(gt_tensor_prev)
        gt_tensor_two_prev_wo_bd = self._remove_boundary(gt_tensor_two_prev)

        losses = (
            # BDF time integrator
            3/2/ self.delta_t * (sr_tensor_wo_bd - 4/3*gt_tensor_prev_wo_bd+ 1/3*gt_tensor_two_prev_wo_bd)
            + self._calculate_spatial_operators(eps, K, b1, b2, sr_tensor, theta)

            # Crank-Nicolson Time integrator
            # self.delta_t * (sr_tensor_wo_bd - gt_tensor_prev_wo_bd)
            # +1/2*( self._calculate_spatial_operators(eps, K, b1, b2, sr_tensor, theta) + self._calculate_spatial_operators(eps, K, b1, b2, gt_tensor_prev, theta))
        )
     
        # print('first part: ', (sr_tensor_wo_bd - 4/3*gt_tensor_prev_wo_bd+ 1/3*gt_tensor_two_prev_wo_bd))
        # print('spatial op: ', self._calculate_spatial_operators(eps, K, b1, b2, sr_tensor))
        
        loss_criteria = nn.MSELoss()
        losses = loss_criteria(losses, torch.zeros_like(losses).to(self.device))

        return losses

    def _calculate_spatial_operators(self, eps: Tensor, K: Tensor, b1:Tensor, b2:Tensor, img: Tensor, theta: Tensor) -> Tensor:
        # img_dx, img_dy = self._calculate_image_derivative(img)
        img_lap = self._calculate_image_laplacian(img)

        img = self._remove_boundary(img)

        spatial_op = (
            - eps * img_lap
            # + b1 * img_dx + b2 * img_dy
            + K*self._nonlinear(phi=img, Theta_=theta)
        )
        # print(F_torch.mse_loss(spatial_op, torch.zeros_like(spatial_op).to(self.device)))
        # non_linear = self._nonlinear(phi=img)
        # print(F_torch.mse_loss(non_linear, torch.zeros_like(non_linear).to(self.device)))

        # print('eps: ', eps)
        # print('img lap: ', img_lap)
        # print('b1: ', b1)
        # print('b2: ', b2)
        # print('img_dx: ', img_dx)
        # print('img_dy: ', img_dy)
        # print('nonlinear: ', self._nonlinear(phi=img))

        return spatial_op

    def _nonlinear(self, phi, Theta_):
        Eps = (1/64)/(2*(2**0.5)*torch.arctanh(torch.Tensor([0.9])))
        Eps = Eps.to(self.device)
        Theta_c = 1.2
       
        opt = 1
        
        return 1/(Eps**2) * self._dFpar(phi, Theta_c, Theta_, opt)

    def _dFpar(self, phi, Theta_c, Theta_, opt):
        if opt == 1:
            
            dF = 0.5*Theta_*(torch.log(1+phi)-torch.log(1-phi))-Theta_c*phi
        return dF
