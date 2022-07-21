import numpy as np
from PIL import Image, ImageFilter
import os
import torch
from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from copy import copy


class ClassSpecificGeneration:
    '''
        produces image maximising given class withh gradient ascent + gaussian blur * weight decay + clipping
    '''
    def __init__(self, model, target):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.use_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.cuda() if self.use_cuda else model
        self.model.eval()
        self.target = target

        # generate a random image (noise) that will be optimised
        self.x = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        self.images = []
    
    def reconstruct_image(self, img):
        recon_img = copy(img.cpu().detach().numpy()[0])

        for channel in range(3):
            recon_img[channel] *= self.std[channel]
            recon_img[channel] += self.mean[channel]
        
        recon_img[recon_img > 1] = 1
        recon_img[recon_img < 0] = 0
        recon_img = np.round(recon_img * 255)

        return np.uint8(recon_img).transpose(1, 2, 0)
    
    def blur_image(self, pil_image, resize=True, blur=None):
        if type(pil_image) != Image.Image:
            try:
                pil_image = Image.fromarray(pil_image)
            except:
                print('check input to the `blur_image`')
        
        if resize:
            pil_image.thumbnail((224, 224))
        if blur is not None:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur))
        
        return pil_image
    
    def preprocess_image(self, pil_img, resize=True, blur=None):
        pil_img = self.blur_image(pil_img, resize, blur)
        arr = np.float32(pil_img).transpose(2, 0, 1)

        for channel, _ in enumerate(arr):
            arr[channel] /= 255
            arr[channel] -= self.mean[channel]
            arr[channel] /= self.std[channel]
        
        tensor = torch.from_numpy(arr).float().unsqueeze(0)

        if self.use_cuda:
            var = Variable(tensor.cuda(), requires_grad=True)
        else:
            var = Variable(tensor, requires_grad=True)
        return var

    def generate(self, iterations=250, blur_frequency=6, blur_radius=0.8, wd=0.05, clip=0.1):
        init_lr = 6

        for i in range(iterations):

            # implement regular gaussian blur to improve the output
            if not i % blur_frequency:
                self.img_ = self.preprocess_image(self.x, resize=False, blur=blur_radius)
            else:
                self.img_ = self.preprocess_image(self.x, resize=False)
            
            if self.use_cuda:
                self.img_ = self.img_.cuda()
            
            optimizer = SGD([self.img_], lr=init_lr, weight_decay=wd)
            output = self.model(self.img_)
            target_loss = -output[0, self.target]

            self.model.zero_grad()
            target_loss.backward()

            if clip:
                nn.utils.clip_grad_norm(self.model.parameters(), clip)
            # update image
            optimizer.step()
            self.x = self.reconstruct_image(self.img_.cpu())
            
            self.images.append(self.x)
        return self.images
