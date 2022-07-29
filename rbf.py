import torch
from torch import nn
import numpy as np


class RBF(nn.Module):
    def __init__(self,
                in_features,
                num_kernels,
                out_features,
                radial_function='gaussian',
                p_norm=2,
                normalization=True,
                init_shape_params=None,
                init_centers_params=None,
                init_weights_params=None,
                const_shape_params=False,
                const_centers_params=False,
                const_weights_params=False):
        super(RBF, self).__init__()

        self.in_features = in_features
        self.num_kernels = num_kernels
        self.out_features = out_features
        self.radial_func = self.basis_functions()[radial_function]
        self.p_norm = p_norm
        self.normalization = normalization
        self.init_shape_params = init_shape_params
        self.init_weights_params = init_weights_params
        self.init_centers_params = init_centers_params
        self.const_shape_params = const_shape_params
        self.const_centers_params = const_centers_params
        self.const_weights_params = const_weights_params

        self.initialize_network()
    
    def initialize_network(self):
        if self.const_weights_params:
            self.weights = nn.Parameter(
                self.init_weights_params, requires_grad=True)
        else:
            self.weights = nn.Parameter(
                torch.zeros(
                    self.out_features,
                    self.num_kernels,
                    dtype=torch.float32))
        
        if self.const_centers_params:
            self.components_centers = nn.Parameter(
                self.init_centers_params, requires_grad=False)
        else:
            self.components_centers = nn.Parameter(
                torch.zeros(
                    self.num_kernels,
                    self.in_features,
                    dtype=torch.float32))

        if self.const_shape_params:
            self.log_shapes = nn.Parameter(
                self.init_shape_params, requires_grad=False)
        else:
            self.log_shapes = nn.Parameter(
                torch.zeros(self.num_kernels, dtype=torch.float32))
        
        self.reset_params()
    
    def reset_params(self,
                     upper_bound_kernels=1.0,
                     std_shape=0.1,
                     gain_weights=1.0):
        if self.init_centers_params is None:
            nn.init.uniform_(
                self.components_centers,
                a=-upper_bound_kernels,
                b=upper_bound_kernels)

        if self.init_shape_params is None:
            nn.init.normal_(self.log_shapes, mean=0.0, std=std_shape)

        if self.init_weights_params is None:
            nn.init.xavier_uniform_(self.weights, gain=gain_weights)
    
    def forward(self, x):
        batch_size = x.size(0)

        centers = self.components_centers.expand(batch_size, self.num_kernels, self.in_features)
        difference = x.view(batch_size, 1, self.in_features) - centers
        rho = self.l_norm(difference, p=self.p_norm)
        eps_rho = self.log_shapes.exp().expand(batch_size, self.num_kernels) * rho

        rbf = self.radial_func(eps_rho)
        if self.normalization:
            rbf = rbf / (1e-9 + rbf.sum(dim=-1)).unsqueeze(-1)
        
        out = self.weights.expand(batch_size, self.out_features, self.num_kernels)
        out = out * rbf.view(batch_size, 1, self.num_kernels)
        return out.sum(dim=-1)
    
    def l_norm(self, x, p):
        return torch.norm(x, p=p, dim=-1)

    def gaussian(self, x):
        return (-x.pow(2)).exp()
    
    def linear(self, x):
        return x
    
    def quadratic(self, x):
        return x.pow(2)

    def multiquadric(self, x):
        return (1 + x.pow(2)).sqrt()
    
    def inverse_multiquadric(self, x):
        return 1 / (1 + x.pow(2)).sqrt()
    
    def inverse_quadratic(self, x):
        return 1 / (1 + x.pow(2))

    def spline(self, x):
        return x.pow(2) * torch.log(x + 1)
    
    def poisson_one(self, x):
        return (x - 1) * torch.exp(-x)
    
    def poisson_two(self, x):
        phi = (x - 2) / 2 * x * torch.exp(-x)
        return phi

    '''
    @property
    def poisson_k(x, k):
        return (x - k) / k * x.pow(k - 1) * torch.exp(-x)
    '''

    def matern32(self, x):
        sqrt3 = np.sqrt(3)
        phi = (1 + sqrt3 * x) * torch.exp(-sqrt3 * x)
        return phi
    
    def matern52(self, x):
        sqrt5 = np.sqrt(5)
        phi = (1 + sqrt5 * x  + 5 / 3 * x.pow(2)) * torch.exp(-sqrt5 * x)
        return phi

    def basis_functions(self):
        bases = {'gaussian': self.gaussian,
             'linear': self.linear,
             'quadratic': self.quadratic,
             'inverse quadratic': self.inverse_quadratic,
             'multiquadric': self.multiquadric,
             'inverse multiquadric': self.inverse_multiquadric,
             'spline': self.spline,
             'poisson one': self.poisson_one,
             'poisson two': self.poisson_two,
             'matern32': self.matern32,
             'matern52': self.matern52}
        return bases

