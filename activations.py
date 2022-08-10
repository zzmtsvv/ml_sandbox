import torch
from torch import nn
from torch.nn import functional as F


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))
    
    @staticmethod
    def backward(ctx, grad_outputs):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_softplus = torch.tanh(F.softplus(x))
        tanh_softplus2 = tanh_softplus * tanh_softplus
        return grad_outputs * (tanh_softplus + x * sigmoid * (1 - tanh_softplus2))


class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SeLU(nn.Module):
    '''
        Implementation of the Scaled Exponential Linear Unit
    '''
    def __init__(self):
        super(SeLU, self).__init__()

        self.lambda_ = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717
    
    def forward(self, x):
        mask = (x > 0).type(x.data.type())

        return self.lambda_ * (x * mask + self.alpha * (torch.exp(x) - 1) * (1 - mask))


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False) -> None:
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return F.relu6(x + 3, self.inplace) / 6


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x * x.sigmoid()


class HardSwish(nn.Module):
    def __init__(self, inplace=False) -> None:
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return x * F.relu6(x + 3, self.inplace) / 6
