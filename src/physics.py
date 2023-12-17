import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
    
class PhysicsLoss(nn.Module):
    """Constructs a physics-based loss function.
     """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> [Tensor]:
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
        device = sr_tensor.device

        # input normalization
        # sr_tensor = self.normalize(sr_tensor)
        # gt_tensor = self.normalize(gt_tensor)

        # sr_tensor and gt_tensor shape: (batch_size, 3, high_res_height, high_res_width)
        u = sr_tensor[:, 0, :, :]
        v = sr_tensor[:, 1, :, :]
        w = sr_tensor[:, 2, :, :]
        # print(u.dtype) # --> torch.float16

        # define the Scharr kernels
        kernel_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float16).view((1, 1, 3, 3)).to(device)
        kernel_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float16).view((1, 1, 3, 3)).to(device)

        du_x = F_torch.conv2d(u.unsqueeze(1), kernel_x, padding=1)
        dv_y = F_torch.conv2d(v.unsqueeze(1), kernel_y, padding=1)
        # dw_z

        # conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        # conv_x.weight = nn.Parameter(kernel_x)

        # conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        # conv_y.weight = nn.Parameter(kernel_y)

        # grad_u_x = conv_x(u.unsqueeze(1))
        # grad_u_y = conv_y(u.unsqueeze(1))
        # grad_u = torch.sqrt(torch.pow(grad_u_x, 2) + torch.pow(grad_u_y, 2))

        # grad_v_x = conv_x(v.unsqueeze(1))
        # grad_v_y = conv_y(v.unsqueeze(1))
        # grad_v = torch.sqrt(torch.pow(grad_v_x, 2) + torch.pow(grad_v_y, 2))

        # grad_w_x = conv_x(w.unsqueeze(1))
        # grad_w_y = conv_y(w.unsqueeze(1))
        # grad_w = torch.sqrt(torch.pow(grad_w_x, 2) + torch.pow(grad_w_y, 2))

        # s = (grad_u_x + grad_v_y) / 1 # TO DO: find the factor
        
        s = (du_x + dv_y) / 1 # TO DO: find the factor
        s = s.reshape(-1)

        losses = torch.mean(torch.square(s))

        losses = losses.to(device)

        return losses

def calculate_pd(u, v):
    device = u.device
    # sr_tensor and gt_tensor shape: (batch_size, 3, high_res_height, high_res_width)
    # u = sr_tensor[:, 0, :, :]
    # v = sr_tensor[:, 1, :, :]
    # w = sr_tensor[:, 2, :, :]
    # print(u.dtype) # --> torch.float16

    # define the Scharr kernels
    kernel_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32).view((1, 1, 3, 3)).to(device)
    kernel_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32).view((1, 1, 3, 3)).to(device)

    du_x = F_torch.conv2d(u.unsqueeze(1), kernel_x, padding=1)
    du_y = F_torch.conv2d(u.unsqueeze(1), kernel_y, padding=1)
    dv_x = F_torch.conv2d(v.unsqueeze(1), kernel_x, padding=1)
    dv_y = F_torch.conv2d(v.unsqueeze(1), kernel_y, padding=1)
    
    # s = (du_x + dv_y) / 1 # TO DO: find the factor
    # s = s.reshape(-1)

    # losses = torch.mean(torch.square(s))

    # losses = losses.to(device)

    return du_x, du_y, dv_x, dv_y

def calculate_div_V_product(u, v, w, xderi_weights, yderi_weights, zderi_weights):
    # image input shape in pytorch: (batch_size, channel, height, width)
    # so, squeeze, unsqueeze are done on axis 1

    # Equation 1
    u_x = F_torch.conv2d(u.unsqueeze(1), xderi_weights, padding='same').squeeze(1)
    u_y = F_torch.conv2d(u.unsqueeze(1), yderi_weights, padding='same').squeeze(1)
    u_z = F_torch.conv2d(u.permute(1, 0, 2).unsqueeze(1), zderi_weights, padding='same').squeeze(1).permute(1, 0, 2)
    u_div = u*u_x + v*u_y + w*u_z

    # Equation 2
    v_x = F_torch.conv2d(v.unsqueeze(1), xderi_weights, padding='same').squeeze(1) 
    v_y = F_torch.conv2d(v.unsqueeze(1), yderi_weights, padding='same').squeeze(1)
    v_z = F_torch.conv2d(v.permute(1, 0, 2).unsqueeze(1), zderi_weights, padding='same').squeeze(1).permute(1, 0, 2)
    v_div = u*v_x + v*v_y + w*v_z

    # Equation 3
    w_x = F_torch.conv2d(w.unsqueeze(1), xderi_weights, padding='same').squeeze(1)
    w_y = F_torch.conv2d(w.unsqueeze(1), yderi_weights, padding='same').squeeze(1) 
    w_z = F_torch.conv2d(w.permute(1, 0, 2).unsqueeze(1), zderi_weights, padding='same').squeeze(1).permute(1, 0, 2)
    w_div = u*w_x + v*w_y + w*w_z
    
    return u_div, v_div, w_div

def calculate_laplacian(u, v, w, xxderi_weights, yyderi_weights, zzderi_weights):

    # Equation 1
    u_xx = F_torch.conv2d(u.unsqueeze(1), xxderi_weights, padding='same').squeeze(1) 
    u_yy = F_torch.conv2d(u.unsqueeze(1), yyderi_weights, padding='same').squeeze(1)
    u_zz = F_torch.conv2d(u.permute(1, 0, 2).unsqueeze(1), zzderi_weights, padding='same').squeeze(1).permute(1, 0, 2)
    u_lap = u_xx + u_yy + u_zz

    # Equation 2
    v_xx = F_torch.conv2d(v.unsqueeze(1), xxderi_weights, padding='same').squeeze(1)
    v_yy = F_torch.conv2d(v.unsqueeze(1), yyderi_weights, padding='same').squeeze(1) 
    v_zz = F_torch.conv2d(v.permute(1, 0, 2).unsqueeze(1), zzderi_weights, padding='same').squeeze(1).permute(1, 0, 2)
    v_lap = v_xx + v_yy + v_zz
    
    # Equation 3
    w_xx = F_torch.conv2d(w.unsqueeze(1), xxderi_weights, padding='same').squeeze(1)
    w_yy = F_torch.conv2d(w.unsqueeze(1), yyderi_weights, padding='same').squeeze(1)
    w_zz = F_torch.conv2d(w.permute(1, 0, 2).unsqueeze(1), zzderi_weights, padding='same').squeeze(1).permute(1, 0, 2)
    w_lap = w_xx + w_yy + w_zz
    
    return u_lap, v_lap, w_lap