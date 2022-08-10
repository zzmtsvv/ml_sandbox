from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class OneCycleLR(_LRScheduler):
    def __init__(self, optimizer, epochs=800, min_lr=0.05, max_lr=1.0):
        super().__init__(optimizer)

        half_epochs = epochs // 2
        decay_epochs = epochs // 20

        lr_grow = np.linspace(min_lr, max_lr, half_epochs)
        lr_down = np.linspace(max_lr, min_lr, half_epochs - decay_epochs)
        lr_decay = np.linspace(min_lr, min_lr * 0.01, decay_epochs)
        self.lrs = np.concatenate((lr_grow, lr_down, lr_decay)) / max_lr
    
    def get_lr(self):
        return [base_lr * self.lrs[self.last_epoch] for base_lr in self.base_lrs]


class CosineAnnealingLRWithDecay(_LRScheduler):
    '''cosine annealing schedule'''
    def __init__(self, optimizer, T_max=80, gamma=0.999, eta_min=0, last_epoch=-1):
        super(CosineAnnealingLRWithDecay, self).__init__(optimizer, last_epoch)

        self.gamma = gamma
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self):
        def compute_lr(base_lr):
            gamma_factor = base_lr * int(np.power(self.gamma, self.last_epoch)) - self.eta_min
            T_max_factor = 1 + float(np.cos(np.pi * self.last_epoch / self.T_max))
            return self.eta_min + gamma_factor * T_max_factor / 2
        
        return [compute_lr(base_lr) for base_lr in self.base_lrs]
