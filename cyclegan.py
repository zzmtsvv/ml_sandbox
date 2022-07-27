# remake from github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from itertools import chain
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import os


def weights_init_normal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm2d' in classname:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


class HorsesZebrasDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transforms = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_X = sorted(glob.glob(os.path.join(root, f'{mode}s/X') + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(root, f'{mode}s/Y') + '/*.*'))
        self.length_X = len(self.files_X)
        self.length_Y = len(self.files_Y)
    
    def __getitem__(self, index):
        item_X = self.transforms(Image.open(self.files_X[index % self.length_X]))

        item_Y = self.transform(Image.open(self.files_Y[index % self.length_Y]))
        if self.unaligned:
            item_Y = self.transforms(Image.open(self.files_Y[np.random.randint(0, self.length_Y - 1)]))
        
        return {'X': item_X, 'Y': item_Y}
    
    def __len__(self):
        return max(self.length_X, self.length_Y)


class Dict:
    epoch = 0 # starting epoch
    n_epochs = 200
    batch_size = 1
    root = ''
    lr = 0.0002
    decay_epoch = 100
    size = 256
    input_channels = 3
    output_channels = 3
    n_cpu = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        to_return = []
        for elem in data.data:
            elem = torch.unsqueeze(elem, 0)
            if len(self.data) < self.max_size:
                self.data.append(elem)
                to_return.append(elem)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = elem
                else:
                    to_return.append(elem)
        return Variable(torch.cat(to_return))


class LambdaRL:
    def __init__(self, epochs, offset, decay_start_epoch):
        assert epochs > decay_start_epoch
        self.n_epochs = epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        res = 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
        return res


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, inp_channels, out_channels, n_resid_blocks=9):
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(inp_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            *self.conv_block(64, 128, 3),
            *self.conv_block(128, 256, 3)
        ]

        for _ in range(n_resid_blocks):
            model.append(ResidualBlock(256))
        
        model += [
            *self.deconv_block(256, 128, 3),
            *self.deconv_block(128, 64, 3),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    
    def conv_block(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        return block

    def deconv_block(self, in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1):
        block = [
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        return block


class Discriminator(nn.Module):
    def __init__(self, inp_channels):
        self.model = nn.Sequential(
            nn.Conv2d(inp_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *self.conv_block(64, 128, 4),
            *self.conv_block(128, 256, 4),
            *self.conv_block(256, 512, 4),
            nn.Conv2d(512, 1, 4, padding=1),
        )
    
    def conv_block(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        return block
    
    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)


class Model:
    def __init__(self):
        self.create_networks()
        self.create_losses()
        self.create_optimizers()
        self.create_datastorages()

        self.transforms_ = [transforms.Resize(int(Dict.size * 1.12), Image.BICUBIC), 
                transforms.RandomCrop(Dict.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
        dataset = HorsesZebrasDataset(
            root=Dict.root,
            transforms_=self.transforms_,
            unaligned=True)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=Dict.batch_size,
            shuffle=True,
            num_workers=Dict.n_cpu)
    
    def create_datastorages(self):
        self.input_X = torch.Tensor(Dict.batch_size, Dict.input_channels, Dict.size, Dict.size)
        self.input_Y = torch.Tensor(Dict.batch_size, Dict.input_channels, Dict.size, Dict.size)
        self.target_real = Variable(torch.Tensor(Dict.batch_size).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(torch.Tensor(Dict.batch_size).fill_(0.0), requires_grad=False)
        self.fake_X_buffer = ReplayBuffer()
        self.fake_Y_buffer = ReplayBuffer()
    
    def create_networks(self):
        self.netG_X2Y = Generator(Dict.input_channels, Dict.output_channels).to(Dict.device)
        self.netF_Y2X = Generator(Dict.input_channels, Dict.output_channels).to(Dict.device)
        self.netD_X = Discriminator(Dict.input_channels).to(Dict.device)
        self.netD_Y = Discriminator(Dict.input_channels).to(Dict.device)

        self.netG_X2Y.apply(weights_init_normal)
        self.netF_Y2X.apply(weights_init_normal)
        self.netD_X.apply(weights_init_normal)
        self.netD_Y.apply(weights_init_normal)
    
    def create_losses(self):
        self.loss_gan = nn.MSELoss()
        self.loss_cycle = nn.L1Loss()
        self.loss_identity = nn.L1Loss()
    
    def create_optimizers(self):
        self.optim_G = optim.Adam(chain(self.netG_X2Y.parameters(), self.netF_Y2X.parameters()), 
                                lr=Dict.lr, 
                                betas=(0.5, 0.999))
        self.optim_D_X = optim.Adam(self.netD_X.parameters(), lr=Dict.lr, betas=(0.5, 0.999))
        self.optim_D_Y = optim.Adam(self.netD_Y.parameters(), lr=Dict.lr, betas=(0.5, 0.999))

        self.lr_scheduler_G = lr_scheduler.LambdaLR(self.optim_G,
                                                    lr_lambda=LambdaRL(Dict.n_epochs,
                                                                        Dict.epoch,
                                                                        Dict.decay_epoch).step)
        self.lr_scheduler_D_X = lr_scheduler.LambdaLR(self.optim_D_X,
                                                lr_lambda=LambdaRL(Dict.n_epochs,
                                                                    Dict.epoch,
                                                                    Dict.decay_epoch).step)
        self.lr_scheduler_D_Y = lr_scheduler.LambdaLR(self.optim_D_Y,
                                                lr_lambda=LambdaRL(Dict.n_epochs,
                                                                    Dict.epoch,
                                                                    Dict.decay_epoch).step)
    
    def cycle_consistency_step(self, real_X, real_Y):
        # generators G_X2Y and F_Y2X
        self.optim_G.zero_grad()

        # Identity loss
        # G_X2Y(real_Y) should be equal to real_Y & F_Y2X(real_X) should be equal to real_X
        pseudoreal_Y = self.netG_X2Y(real_Y)
        loss_identity_Y = 5.0 * self.loss_identity(pseudoreal_Y, real_Y) 
        pseudoreal_X = self.netF_Y2X(real_X)
        loss_identity_X = 5.0 * self.loss_identity(pseudoreal_X, real_X)

        # GAN loss
        fake_Y = self.netG_X2Y(real_X)
        pred_on_fake = self.netD_Y(fake_Y)
        loss_GAN_X2Y = self.loss_gan(pred_on_fake, self.target_real)

        fake_X = self.netF_Y2X(real_Y)
        pred_on_fake = self.netD_X(fake_X)
        loss_GAN_Y2X = self.loss_gan(pred_on_fake, self.target_real)

        # cycle loss
        recovered_X = self.netF_Y2X(fake_Y)
        loss_cycle_XYX = 10.0 * self.loss_cycle(recovered_X, real_X)

        recovered_Y = self.netG_X2Y(fake_X)
        loss_cycle_YXY = 10.0 * self.loss_cycle(recovered_Y, real_Y)

        loss_G = loss_identity_X + loss_identity_Y
        loss_G = loss_G + loss_GAN_X2Y + loss_GAN_Y2X
        loss_G = loss_G + loss_cycle_XYX + loss_cycle_YXY
        loss_G.backward()
        self.optim_G.step()

        return fake_X, fake_Y
    
    def discriminator_x_step(self, real_X, fake_X):
        # Discriminator X
        self.optim_D_X.zero_grad()

        pred_on_real = self.netD_X(real_X)
        loss_D_real = self.loss_gan(pred_on_real, self.target_real)

        fake_X = self.fake_X_buffer.push_and_pop(fake_X)
        pred_on_fake = self.netD_X(fake_X.detach())
        loss_D_fake = self.loss_gan(pred_on_fake, self.target_fake)

        loss_D_X = 1 / 2 * (loss_D_real + loss_D_fake)
        loss_D_X.backward()
        self.optim_D_X.step()
    
    def discriminator_y_step(self, real_Y, fake_Y):
        self.optim_D_Y.zero_grad()

        pred_on_real = self.netD_Y(real_Y)
        loss_D_real = self.loss_gan(pred_on_real, real_Y)

        fake_Y = self.fake_Y_buffer.push_and_pop(fake_Y)
        pred_on_fake = self.netD_Y(fake_Y.detach())
        loss_D_fake = self.loss_gan(pred_on_fake, self.target_fake)

        loss_D = 1 / 2 * (loss_D_real + loss_D_fake)
        loss_D.backward()

        self.optim_D_Y.step()

    def update_batch(self, real_X, real_Y):
        fake_X, fake_Y = self.cycle_consistency_step(real_X, real_Y)

        self.discriminator_x_step(real_X, fake_X)
        self.discriminator_y_step(real_Y, fake_Y)


    def train(self):
        for epoch in range(Dict.epoch, Dict.n_epochs):
            for i, batch in enumerate(self.dataloader):
                real_X = Variable(self.input_X.copy_(batch['X']))
                real_Y = Variable(self.input_Y.copy_(batch['Y']))

                self.update_batch(real_X, real_Y)
            
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_X.step()
            self.lr_scheduler_D_Y.step()


