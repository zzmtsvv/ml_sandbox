from unittest import result
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
    
    def forward(self, x):
        '''
        x - latents
        '''
        x = x.permute(0, 2, 3, 1).contiguous() # [B x D x H x W] -> [B x H x W x D]
        latents_shape = x.shape

        flattened_latents = x.view(-1, self.D)

        # L2 distance between latents and embeddings
        dist = torch.sum(flattened_latents ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True)
        dist = dist - 2 * torch.matmul(flattened_latents, self.embedding.weight.t())  # [BHW, K]

        # discretization bottleneck
        encoding_indexes = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # one-hot encoding
        device = x.device
        encoded = torch.zeros(encoding_indexes.size(0), self.K, device=device).scatter_(1, encoding_indexes, 1)  # [BHW, K]

        # quantize the latents
        quantized_latents = torch.matmul(encoded, self.embedding.weight).view(latents_shape)  # [B x H x W x D]

        # VQ losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), x)
        emb_loss = F.mse_loss(quantized_latents, x.detach())
        # stopgradient operation is equivalent to detaching the tensor from the current computational graph
        # (considered as a constant, do not requires the gradient)
        vq_loss = commitment_loss + self.beta * emb_loss

        # residuals back to quantized part
        quantized_latents = x + (quantized_latents - x).detach()

        mean_probs = torch.mean(encoded, dim=0)
        perplexity = torch.exp(-torch.sum(mean_probs * torch.log(mean_probs + 1e-10)))

        # [B x D x H x W]
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss, perplexity, encoded
   
        
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        return x + self.resblock(x)


class VQVAE(nn.Module):
    '''
    https://arxiv.org/abs/1711.00937
    '''
    def __init__(self, in_channels, embedding_dim, num_embeddings, hidden_dims=None, beta=0.25):
        super(VQVAE, self).__init__(in_channels=in_channels)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [128, 256]
        
        # encoder
        modules = self.build_encoder(hidden_dims)
        self.encoder = nn.Sequential(*modules)
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.beta)

        hidden_dims.reverse()

        # decoder
        modules = self.build_decoder(hidden_dims)
        self.decoder = nn.Sequential(*modules)
        
        
    def build_encoder(self, hidden_dims):
        modules = []
        in_channels = self.in_channels

        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            in_channels = dim
        
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )

        for _ in range(6):
            modules.append(Residual(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, self.embedding_dim, kernel_size=1, stride=1),
            nn.LeakyReLU()
        ))

        return modules
    
    def build_decoder(self, hidden_dims):
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(self.embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        ))

        for _ in range(6):
            modules.append(Residual(hidden_dims[0], hidden_dims[0]))
        modules.append(nn.LeakyReLU())

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ))
    
    def encode(self, x):
        '''
        :param x: input tensor to encoder [N x C x H x W]
        :return: list of latent codes
        '''
        return [self.encoder(x)]
    
    def decode(self, z):
        '''
        maps latent codes onto the image space
        :param z: Tensor [B x D x H x W]
        :return: Tensor [B x C x H x W]
        '''
        return self.decoder(z)
    
    def generate(self, x):
        '''
        returns the reconstructed image [B x C x H x W]
        '''
        return self.forward(x)[0]
    
    def forward(self, x):
        encoded = self.encoder(x)[0]
        quantized, vq_loss, perplexity, encoded = self.vq_layer(encoded)
        return [self.decode(quantized), x, vq_loss, perplexity, encoded]
    
    def loss_function(self, *args):
        '''
        see self.forward() for details of args
        '''
        reconstructed, x, vq_loss, perplexity, _ = args

        reconstruction_loss = F.mse_loss(reconstructed, x)
        loss = reconstruction_loss + vq_loss
        return loss, reconstruction_loss, perplexity
    
    def sample(self, num_samples, device):
        '''
        samples from the latent space and maps to the image space
        '''
        z = torch.rand(num_samples, 1, self.vq_layer.K, self.vq_layer.D)
        z.to(device)

        quantized_latents, _, _, _ = self.vq_layer(z)
        return self.decode(quantized_latents)
