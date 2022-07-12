import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class VAE(nn.Module):
    '''
    https://arxiv.org/abs/1312.6114
    '''
    def __init__(self, in_channels, latent_dim, hidden_dims=None):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [np.power(2, i) for i in range(5, 10)]
        
        modules = self.build_encoder(hidden_dims)
        self.encoder = nn.Sequential(*modules)
        self.fc_mean = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_variance = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        modules = self.build_decoder(hidden_dims)
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def build_encoder(self, hidden_dims):
        modules = []
        in_channels = self.in_channels

        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU())
            )
            in_channels = dim

        return modules
    
    def build_decoder(self, hidden_dims):
        modules = []

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            ))
        
        return modules


    def generate(self, x):
        '''
        returns the reconstructed image
        :param x: tensor [B x C x H x W]
        :return: tensor [B x C x H x W]
        '''
        return self.forward(x)[0]
    
    def sample(self, num_samples, device):
        '''
        Samples from the latent space and maps to image space map
        '''
        z = torch.randn(num_samples, self.latent_dim)
        z.to(device)

        return self.decode(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]
    
    def reparameterize(self, mu, log_var):
        '''
        Reparameterization trick to sample N(mu, var) from N(0, 1)
        :param mu: Mean of the latent Gaussian [B x D]
        :param log_var: standard deviation of the latent Gaussian [B x D]
        '''
        std = torch.exp(1 / 2 * log_var)
        eps = torch.rand_like(std)
        return eps * std + mu
    
    def decode(self, z):
        '''
        reconstruct the image given the latent embedding
        :param z: latent codes [B x D]
        :return: tensor [B x C x H x W]
        '''
        res = self.decoder_input(z).view(-1, 512, 2, 2)
        res = self.decoder(res)
        return self.final_layer(res)
        

    def encode(self, x):
        '''
        Encodes the input by passing through the encoder network and returns latent codes
        :param x: (Tensor) input tensor to encoder [N x C x H x W]
        :return: list of parameters of latent Gaussian distribution
        '''
        res = torch.flatten(self.encoder(x), start_dim=1)

        mu = self.fc_mean(res)
        log_variance = self.fc_variance(res)

        return [mu, log_variance]
    
    def loss_function(self, *args, weight=None):
        '''
        Computes the VAE loss function (KL divergence)
        '''
        
        # see self.forward() for details of unpacking the args
        x_hat, x, mu, log_var = args

        # KL divergence weight
        if weight is None:
            weight = 1
        
        reconstruction_loss = F.mse_loss(x_hat, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - torch.square(mu) - log_var.exp(), dim=1), dim=0)
        loss = reconstruction_loss + weight * kld_loss

        return loss, reconstruction_loss, kld_loss
