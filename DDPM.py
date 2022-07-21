import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import numpy as np
from functools import partial
from tqdm.auto import tqdm
from einops import rearrange
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize


IMAGE_SIZE = 128
TIMESTEPS = 200

transform = Compose([
    Resize(IMAGE_SIZE),
    CenterCrop(IMAGE_SIZE),
    ToTensor(),
    Lambda(lambda x: 2 * x - 1)])

reverse_transform = Compose([
     Lambda(lambda x: (x + 1) / 2),
     Lambda(lambda x: x.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda x: x * 255.),
     Lambda(lambda x: x.numpy().astype(np.uint8)),
     ToPILImage(),
])

def launch(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def beta_schedule(timesteps, s=0.008, mode='linear'):
    if mode == 'cosine':
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    
    beta_start = 0.0001
    beta_end = 0.02

    if mode == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps)

    if mode == 'quadratic':
        betas = torch.linspace(np.sqrt(beta_start), np.sqrt(beta_end), timesteps) ** 2
    
    if mode == 'sigmoid':
        betas = torch.linspace(-6, 6, timesteps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    
    else:
        raise NotImplementedError()
    
    return betas

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

betas = beta_schedule(timesteps=TIMESTEPS)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_inverted_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x0.shape)

    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_img(x0, t):
    return reverse_transform(q_sample(x0, t).squeeze())

def p_loss(model, x0, t, noise=None, loss_type='huber'):
    if noise is None:
        noise = torch.randn_like(x0)
    x_noisy = q_sample(x0, t, noise)
    prediction = model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, prediction)
    if loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, prediction)
    if loss_type == 'l2':
        loss = F.mse_loss(noise, prediction)
    else:
        raise NotImplementedError()
    
    return loss

@torch.no_grad()
def p_sample(model, x, t, t_idx):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_inverted_alphas_t = extract(sqrt_inverted_alphas, t, x.shape)

    mean = sqrt_inverted_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    if not t_idx:
        return mean
    
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    noise = torch.rand_like(x)
    return mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    bs = shape[0]
    img = torch.randn(shape, device=device)
    history = []

    for i in tqdm(range(TIMESTEPS, -1, -1), desc='sampling loop in time', total=TIMESTEPS):
        img = p_sample(model, img, torch.full((bs,), i, device=device, dtype=torch.long), i)
        history.append(img.cpu().numpy())
    return history

@torch.no_grad()
def sample(model, img_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, img_size, img_size))


class PositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super(PositionEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, time):
        device = time.device
        half = self.embedding_dim // 2
        
        embeddings = np.log(10000) / (half - 1)
        embeddings = torch.exp(torch.arange(half, device) * (-1) * embeddings)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CustomResidual(nn.Module):
    def __init__(self, fn):
        super(CustomResidual, self).__init__()

        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)


class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()

        self.dim = dim
        self.fn = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
    
    def forward(self, x):
        return self.fn(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()

        self.dim = dim
        self.fn = nn.Conv2d(dim, dim, 4, 2, 1)
    
    def forward(self, x):
        return self.fn(x)


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.activation = nn.SiLU()
    
    def forward(self, x, scale_n_shift=None):
        x = self.norm(self.conv(x))

        if scale_n_shift is not None:
            scale, shift = scale_n_shift
            x = x * (scale + 1) + shift
        return self.activation(x)


class ResNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *args, time_embedding_dim=None, groups=8):
        super(ResNetBlock, self).__init__()

        if time_embedding_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embedding_dim, dim_out))
        else:
            self.mlp = None
        
        self.block1 = ConvBlock(dim_in, dim_out, groups)
        self.block2 = ConvBlock(dim_out, dim_out, groups)
        self.residual = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
    
    def forward(self, x, time_embedding=None):
        hidden = self.block1(x)

        if self.mlp is not None and time_embedding is not None:
            time_embedding = self.mlp(time_embedding)
            hidden = rearrange(time_embedding, 'b c -> b c 1 1') + hidden  # unsqueeze 3rd and 4th dimensions
        
        return self.block2(hidden) + self.residual(x)


class ConvNextBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *args, time_embedding_dim=None, mult=2, norm=True):
        super(ConvNextBlock, self).__init__()

        if time_embedding_dim is not None:
            self.mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_embedding_dim, dim_in))
        else:
            self.mlp = None
        
        self.conv = nn.Conv2d(dim_in, dim_in, 7, padding=3, groups=dim_in)

        self.NN = nn.Sequential(
            nn.GroupNorm(1, dim_in) if norm else nn.Identity(),
            nn.Conv2d(dim_in, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1))
        
        self.residual = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
    
    def forward(self, x, time_embedding=None):
        hidden = self.conv(x)

        if self.mlp is not None and time_embedding is not None:
            time_embedding = self.mlp(time_embedding)
            hidden = rearrange(time_embedding, 'b c -> b c 1 1') + hidden  # unsqueeze 3rd anf 4th dimensions
        
        return self.NN(hidden) + self.residual(x)


class Attention(nn.Module):
    '''
    realization of the Attention mechanism with the einops module
    '''
    def __init__(self, dim_in, heads=4, dim_head=32):
        super(Attention, self).__init__()

        self.scale = 1 / np.sqrt(dim_head)
        self.heads = heads
        hidden_dim = dim_head * heads

        self.query_key_value = nn.Conv2d(dim_in, hidden_dim * 3, 1, bias=False)
        self.out_ = nn.Conv2d(hidden_dim, dim_in, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.query_key_value(x).chunk(3, dim=1)
        query, key, value = map(
            lambda z: rearrange(z, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv
        )
        query = query * self.scale

        similarity = torch.einsum('b h d i, b h d j -> b h i j', query, key)
        similarity = similarity - similarity.amax(dim=-1, keepdim=True).detach()  # avoiding vanishing gradients
        attention = similarity.softmax(dim=-1)

        out = torch.einsum('b h i j, b h d j -> b h i d', attention, value)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.out_(out)


class LinearAttention(nn.Module):
    '''
    realization of the Linear Attention mechanism with the einops module
    '''
    def __init__(self, dim_in, heads=4, dim_head=32):
        super(Attention, self).__init__()

        self.scale = 1 / np.sqrt(dim_head)
        self.heads = heads
        hidden_dim = dim_head * heads

        self.query_key_value = nn.Conv2d(dim_in, hidden_dim * 3, 1, bias=False)
        self.out_ = nn.Sequential(
            nn.Conv2d(hidden_dim, dim_in, 1),
            nn.GroupNorm(1, dim_in))
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.query_key_value(x).chunk(3, dim=1)
        query, key, value = map(
            lambda z: rearrange(z, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv
        )
        query = query.softmax(dim=-2)
        key = key.softmax(dim=-1)
        query = query * self.scale
        context = torch.einsum('b h i j, b h d j -> b h i d', key, value)

        out = torch.einsum('b h d i, b h d j -> b h i j', context, query)
        out = out - out.amax(dim=-1, keepdim=True).detach()  # avoiding vanishing gradients
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h=self.heads, x=h, y=w)
        return self.out_(out)


class PreNorm(nn.Module):
    def __init__(self, dim_in, fn):
        super(PreNorm, self).__init__()

        self.fn = fn
        self.norm_ = nn.GroupNorm(1, dim_in)
    
    def forward(self, x):
        return self.fn(self.norm_(x))


class Unet(nn.Module):
    def __init__(self, dim_in, init_dim=None, dim_out=None, dim_mults=(1, 2, 4, 8),
                channels=3, is_time_embedding=True, resnet_block_grps=8, is_convnext=True,
                convnext_mult=2):
        super(Unet, self).__init__()
        
        self.channels = channels
        init_dim = launch(init_dim, dim_in // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        dimensions = [init_dim, *map(lambda dim: dim_in * dim, dim_mults)]
        ins_outs = list(zip(dimensions[:-1], dimensions[1:]))
        block = partial(ResNetBlock, groups=resnet_block_grps)
        time_dim = None
        self.time_mlp = None

        if is_convnext:
            block = partial(ConvNextBlock, mult=convnext_mult)
        
        if is_time_embedding:
            time_dim = dim_in * 4
            self.time_mlp = nn.Sequential(
                PositionEmbeddings(dim_in),
                nn.Linear(dim_in, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        self.downsample = nn.ModuleList([])
        self.upsample = nn.ModuleList([])
        resolutions = len(ins_outs)

        for idx, (in_dim, out_dim) in enumerate(ins_outs):
            is_last = idx >= resolutions - 1
            self.downsample.append(
                nn.ModuleList(
                    [
                        block(in_dim, out_dim, time_embedding_dim=time_dim),
                        block(out_dim, out_dim, time_embedding_dim=time_dim),
                        CustomResidual(PreNorm(in_dim, LinearAttention(in_dim))),
                        Downsample(in_dim) if not is_last else nn.Identity()
                    ]))
        middle_dim = dimensions[-1]
        self.mid_block1 = block(middle_dim, middle_dim, time_embedding_dim=time_dim)
        self.mid_attention = CustomResidual(PreNorm(middle_dim, Attention(middle_dim)))
        self.mid_block2 = block(middle_dim, middle_dim, time_embedding_dim=time_dim)

        for idx, (in_dim, out_dim) in enumerate(reversed(ins_outs[1:])):
            is_last = idx >= resolutions - 1
            self.upsample.append(
                nn.ModuleList(
                    [
                        block(out_dim * 2, in_dim, time_embedding_dim=time_dim),
                        block(in_dim, in_dim, time_embedding_dim=time_dim),
                        CustomResidual(PreNorm(in_dim, LinearAttention(in_dim))),
                        Upsample(in_dim) if not is_last else nn.Identity()
                    ]))
        dim_out = launch(dim_out, channels)
        self.final_conv = nn.Sequential(
            block(dim_in, dim_in), nn.Conv2d(dim_in, dim_out, 1)
        )

    
    def forward(self, x, time):
        x = self.init_conv(x)
        t = None
        if self.time_mlp is not None:
            t = self.time_mlp(time)
        h = []

        for block1, block2, attention, downsample in self.downsample:
            x = attention(block2(block1(x, t), t))
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attention(x)
        x = self.mid_block2(x, t)

        for block1, block2, attention, upsample in self.upsample:
            x = torch.cat((x, h.pop()), dim=1)
            x = attention(block2(block1(x, t), t))
            x = upsample(x)
        return self.final_conv(x)
