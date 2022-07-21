'''
code has been run in google colab
'''

import torch
from torchvision import transforms
from PIL import Image
from torch import nn
from torchvision import models
from torch import optim
from torchvision.utils import save_image
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def image_loader(im_path):
  img = Image.open(im_path)
  transform = transforms.Compose([
                                  transforms.Resize((512, 512)),
                                  transforms.ToTensor()])
  img = transform(img).unsqueeze(0)
  return img.to(device, torch.float)

original_image = image_loader('/content/оригинал.jpeg')
style_image = image_loader('/content/photo_2022-07-15 20.58.10.jpeg')

img = original_image.clone().requires_grad_(True)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.req_features = [0, 5, 10, 19, 28]
    self.model = models.vgg19(pretrained=True).features[:29]
  
  def forward(self, x):
    features = []

    for idx, layer in enumerate(self.model):
      x = layer(x)
      if idx in self.req_features:
        features.append(x)
    return features

class Trainer:
  def __init__(self, original_img, style_img, gen_img):
    self.orig_img = original_img
    self.style_img = style_img
    self.img = gen_img

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    self.model = Net().to(device).eval()
    self.epochs = 10000
    self.lr = 0.004
    self.alpha = 8
    self.beta = 70
    self.optimizer = optim.Adam([self.img], lr=self.lr)
  
  def content_loss(self, gen_feat, orig_feat):
    return F.mse_loss(gen_feat, orig_feat)
  
  def style_loss(self, gen_feat, style_feat):
    b, c, h, w = gen_feat.shape

    G = torch.mm(gen_feat.view(c, h * w), gen_feat.view(c, h * w).t())
    A = torch.mm(style_feat.view(c, h * w), style_feat.view(c, h * w).t())
    loss = F.mse_loss(G, A)
    return loss
  
  def loss(self, gen_features, orig_features, style_features):
    style_loss, content_loss = 0, 0

    for gen, orig, style in zip(gen_features, orig_features, style_features):
      content_loss += self.content_loss(gen, orig)
      style_loss += self.style_loss(gen, style)
    
    total_loss = self.alpha * content_loss + self.beta * style_loss
    return total_loss
  
  def train(self):
    for e in range(1, self.epochs + 1):
      gen_features = self.model(self.img)
      orig_features = self.model(self.orig_img)
      style_features = self.model(self.style_img)

      total_loss = self.loss(gen_features, orig_features, style_features)
      self.optimizer.zero_grad()
      total_loss.backward()
      self.optimizer.step()

      if not e % 100:
        print(total_loss)

        save_image(self.img, f'/content/generated0/gen_{e}_epoch.png')

t = Trainer(original_image, style_image, img)

t.train()
