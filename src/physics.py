import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from functools import reduce

def calculate_image_derivative(img: Tensor) -> [Tensor, Tensor]:        
    # Filter kernels
    k = Tensor([1., 3.5887, 1.]) # or [3., 10., 3.] or [17., 61., 17.]
    d = Tensor([1., 0., -1.])
    gy = torch.outer(k, d)
    gx = gy.transpose(0, 1)

    coeff = - 5.645298778954285 # empirically found through comparing with analytical solutions

    img_dx = coeff * F_torch.conv2d(img, gx.view(1, 1, 3, 3).to(img.device))
    img_dy = coeff * F_torch.conv2d(img, gy.view(1, 1, 3, 3).to(img.device))

    return img_dx, img_dy

def calculate_image_laplacian(img: Tensor) -> Tensor:

    # Filter kernel
    g = torch.Tensor([[0., 1., 0.],
                    [1., 4., 1.],
                    [0., 1., 0.]])

    coeff = - 9.880939350316519 # empirically found through comparing with analytical solutions
    
    img_lap = coeff * F_torch.conv2d(img, g.view(1, 1, 3, 3).to(img.device))
        
    return img_lap

def remove_boundary(tensor: Tensor) -> Tensor:
        # (batch, channel, 64, 64) -> (batch, channel, 62, 62)
        return tensor[:, :, 1:-1, 1:-1]

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
        sr_tensor_wo_bd = remove_boundary(sr_tensor)
        gt_tensor_prev_wo_bd = remove_boundary(gt_tensor_prev)
        gt_tensor_two_prev_wo_bd = remove_boundary(gt_tensor_two_prev)

        # contributed by Pouria Behnoudfar
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

        loss_criteria = nn.MSELoss()
        losses = loss_criteria(losses, torch.zeros_like(losses).to(self.device))

        return losses

    def _calculate_spatial_operators(self, eps: Tensor, K: Tensor, b1:Tensor, b2:Tensor, img: Tensor) -> Tensor:
        # input normalization
        # sr_tensor = self.normalize(sr_tensor)
        # gt_tensor = self.normalize(gt_tensor)

        img_dx, img_dy = calculate_image_derivative(img)
        img_lap = calculate_image_laplacian(img)

        img = remove_boundary(img)

        # contributed by Pouria Behnoudfar
        spatial_op = (
            - eps * img_lap
            + b1 * img_dx + b2 * img_dy
            - K * torch.pow(img, 2) * (1-img)
        )
        
        return spatial_op
    
    def sanity_check(self, img):
        img_dx, img_dy = self._calculate_image_derivative(img)
        img_dx_dx, _ = self._calculate_image_derivative(img_dx)
        _, img_dy_dy = self._calculate_image_derivative(img_dy)

        left = img_dx_dx + img_dy_dy
        print(left, left.shape)
        print('---')

        img_lap = calculate_image_laplacian(img)
        img_lap = remove_boundary(img_lap)
        print(img_lap, img_lap.shape)
        assert left.shape == img_lap.shape

        print('Differences:')
        print(left - img_lap)

        print('---')
        print(torch.allclose(img_dx_dx + img_dy_dy, img_lap))

    
class PhysicsLossImageBoundary(nn.Module):
    """Constructs a physics-based loss function for the boundary of the image.
     """

    def __init__(self, boundary_type: str) -> None:
        super().__init__()
        self.boundary_type = boundary_type
        assert self.boundary_type in ['Dirichlet', 'Periodic', 'Neumann'], "Boundary type must be one of ['Dirichlet', 'Periodic', 'Neumann']"

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        
        # get device
        self.device = sr_tensor.device

        # input normalization
        # sr_tensor = self.normalize(sr_tensor)

        if self.boundary_type == 'Dirichlet':
            ## Dirichlet boundary conditions
            losses_top = F_torch.mse_loss(sr_tensor[:, :, 0, :], gt_tensor[:, :, 0, :]).to(self.device)
            losses_bottom = F_torch.mse_loss(sr_tensor[:, :, -1, :], gt_tensor[:, :, -1, :]).to(self.device)
            losses_left = F_torch.mse_loss(sr_tensor[:, :, 1:-1, 0], gt_tensor[:, :, 1:-1, 0]).to(self.device) # due to overalp, remove 1 pixel from top and bottom
            losses_right = F_torch.mse_loss(sr_tensor[:, :, 1:-1, -1], gt_tensor[:, :, 1:-1, -1]).to(self.device)
            
            losses = losses_top + losses_bottom + losses_left + losses_right
        
        elif self.boundary_type == 'Periodic':
            ## Periodic boundary conditions
            losses_top_bottom = F_torch.mse_loss(sr_tensor[:, :, 0, :], sr_tensor[:, :, -1, :]).to(self.device)
            losses_left_right = F_torch.mse_loss(sr_tensor[:, :, 1:-1, 0], sr_tensor[:, :, 1:-1, -1]).to(self.device) # due to overalp, remove 1 pixel from top and bottom

            losses = losses_left_right + losses_top_bottom

        elif self.boundary_type == 'Neumann':
            ## Neumann boundary conditions
            losses_top = F_torch.mse_loss(sr_tensor[:, :, 0, :], sr_tensor[:, :, 1, :]).to(self.device)
            losses_bottom = F_torch.mse_loss(sr_tensor[:, :, -1, :], sr_tensor[:, :, -2, :]).to(self.device)
            losses_left = F_torch.mse_loss(sr_tensor[:, :, 1:-1, 0], sr_tensor[:, :, 1:-1, 1]).to(self.device)
            losses_right = F_torch.mse_loss(sr_tensor[:, :, 1:-1, -1], sr_tensor[:, :, 1:-1, -2]).to(self.device)

            losses = losses_top + losses_bottom + losses_left + losses_right

        return losses
    
class PhysicsLossInnerImageAllenCahn(nn.Module):
    """Constructs a physics-based loss function for the inner side of the image (w/o boundary pixels).
     """

    def __init__(self, time_integrator: str) -> None:
        super().__init__()

        # FEM data generation parameters
        self.delta_t = 0.001 / 100 # T/n_samples
        self.time_integrator = time_integrator

        assert self.time_integrator in ['BDF', 'CN', 'EE'], "Time integrator must be one of ['BDF', 'CN', 'EE']"

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

        # remove boundary
        sr_tensor_wo_bd = remove_boundary(sr_tensor)
        gt_tensor_prev_wo_bd = remove_boundary(gt_tensor_prev)
        gt_tensor_two_prev_wo_bd = remove_boundary(gt_tensor_two_prev)

        # contributed by Pouria Behnoudfar
        if self.time_integrator == 'BDF':
            # BDF time integrator
            losses = (
                3/2/ self.delta_t * (sr_tensor_wo_bd - 4/3*gt_tensor_prev_wo_bd+ 1/3*gt_tensor_two_prev_wo_bd)
                + self._calculate_spatial_operators(eps, K, b1, b2, sr_tensor, theta)
            )
        elif self.time_integrator == 'CN':
            # Crank-Nicolson Time integrator
            losses = (
               1/self.delta_t * (sr_tensor_wo_bd - gt_tensor_prev_wo_bd)
                +1/2*( self._calculate_spatial_operators(eps, K, b1, b2, sr_tensor, theta) + self._calculate_spatial_operators(eps, K, b1, b2, gt_tensor_prev, theta))
            )
        elif self.time_integrator == 'EE':
            # Euler Explicit Time integrator
            losses = (
                1/self.delta_t * (sr_tensor_wo_bd - gt_tensor_prev_wo_bd)
                + self._calculate_spatial_operators(eps, K, b1, b2, gt_tensor_prev, theta) 
            )
        
        losses = F_torch.mse_loss(losses, torch.zeros_like(losses).to(self.device))

        return losses

    def _calculate_spatial_operators(self, eps: Tensor, K: Tensor, b1:Tensor, b2:Tensor, img: Tensor, theta: Tensor) -> Tensor:
        # img_dx, img_dy = self._calculate_image_derivative(img)
        img_lap = calculate_image_laplacian(img)

        img = remove_boundary(img)

        # contributed by Pouria Behnoudfar
        spatial_op = (
            - eps * img_lap
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
        # contributed by Pouria Behnoudfar
        Eps = (1/64)/(2*(2**0.5)*torch.arctanh(torch.Tensor([0.9]))) # is 64 fixed or related to any things?
        Eps = Eps.to(self.device)
        Theta_c = 1.2
       
        opt = 1
        
        return 1/(Eps**2) * self._dFpar(phi, Theta_c, Theta_, opt)

    def _dFpar(self, phi, Theta_c, Theta_, opt):
        # contributed by Pouria Behnoudfar
        if opt == 1:           
            dF = 0.5*Theta_*(torch.log(1+phi)-torch.log(1-phi))-Theta_c*phi
        return dF
    

class PhysicsLossInnerImageEriksonJohnson(PhysicsLossInnerImageAllenCahn):
    """Constructs a physics-based loss function for the boundary of the image for Erickson-Johnson data.
     """
    def __init__(self, time_integrator: str = 'BDF') -> None:
        super().__init__(time_integrator)
        self.delta_t = 0.005 # comes from FEM
    
    def _calculate_spatial_operators(self, eps: Tensor, K: Tensor, b1:Tensor, b2:Tensor, img: Tensor, theta: Tensor) -> Tensor:
        img_dx, img_dy = calculate_image_derivative(img)
        img_lap = calculate_image_laplacian(img)

        img = remove_boundary(img)

        # contributed by Pouria Behnoudfar
        spatial_op = (
            - eps * img_lap
            + b1 * img_dx + b2 * img_dy
            + K*img*(img-1)
        )

        return spatial_op
    

class H1Error(nn.Module):
    '''Calculate H1 error between two images
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        assert sr_tensor.size() == gt_tensor.size(), "Tensors must have the same size"

        # get device
        self.device = sr_tensor.device

        sr_dx, sr_dy = calculate_image_derivative(sr_tensor)
        gt_dx, gt_dy = calculate_image_derivative(gt_tensor)
        
        # DX = sr_dx - gt_dx
        # DY = sr_dy - gt_dy
        # h1_error= torch.square(DX) + torch.square(DY)
        # shape = h1_error.shape
        # num_ele = reduce(lambda x, y: x*y, shape)
        # h1_error = torch.sum(h1_error) / num_ele

        h1_error = F_torch.mse_loss(sr_dx, gt_dx) + F_torch.mse_loss(sr_dy, gt_dy)
        # h1_ = F_torch.mse_loss(gt_dx, torch.zeros_like(gt_dx)) + F_torch.mse_loss(gt_dy, torch.zeros_like(gt_dy))

        return h1_error
