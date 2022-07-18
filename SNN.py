import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import f1_score
from tqdm import tqdm
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SeLU(nn.Module):
    def __init__(self):
        super(SeLU, self).__init__()

        self.lambda_ = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717
    
    def forward(self, x):
        mask = (x > 0).type(x.data.type())

        return self.lambda_ * (x * mask + self.alpha * (torch.exp(x) - 1) * (1 - mask))


class AlphaDropout(nn.Module):
    def __init__(self, dropout_rate):
        super(AlphaDropout, self).__init__()

        self.lambda_ = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717
        self.alpha_prime = - self.lambda_ * self.alpha

        self.p = dropout_rate
        self.q = 1 - self.p

        self.a = 1 / np.sqrt(self.q + self.alpha_prime * self.alpha_prime * self.q * self.p)
        # self.b = -self.a * self.p * self.a_prime
    
    def forward(self, x):
        if not self.training:
            return x
        
        mask = torch.bernoulli(torch.ones(x.size()) * self.p)
        x.masked_fill_(Variable(mask.byte().to(device)), self.alpha_prime)
        
        return self.a * (x + 1)


class MLP(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, in_features, out_features, p_drop=0.2, self_normalized=False):
        super(MLP, self).__init__()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)
        if self_normalized:
            self.activation = SeLU()
            self.dropout = AlphaDropout(p_drop)
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            *self.nonlinear_block(in_features, 512),
            *self.nonlinear_block(512, 256),
            nn.Linear(256, out_features)
        )

        if self_normalized:
            for param in self.mlp.parameters():
                # bias
                if len(param.shape) == 1:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
    
    def nonlinear_block(self, in_features, out_features):
        block = [
            nn.Linear(in_features, out_features),
            self.activation,
            self.dropout
        ]
        return block
    
    def forward(self, x):
        return self.mlp(x)


class F1(nn.Module):
    def __init__(self):
        super(F1, self).__init__()

    def forward(self, x, y):
        pred = F.softmax(x, dim=1).argmax(dim=1).cpu().numpy()
        y = y.cpu().numpy()

        return f1_score(y, pred, average='weighted')


def forward_pass(network, data, loss_fn):
    for x, y in data:
        x = x.to(network.device)

        pred = network(x).cpu()
        loss = loss_fn(pred, y)
        yield loss

@torch.enable_grad()
def update(network, data, loss, optimizer):
    network.train()

    errors = []
    for err in forward_pass(network, data, loss):
        errors.append(err.item())
        optimizer.zero_grad()

        err.backward()
        optimizer.step()
    
    return errors

@torch.no_grad()
def evaluate(network, data, metric):
    network.eval()

    result = []
    for res in forward_pass(network, data, metric):
        result.append(res.item())
    
    return np.mean(result).item()

def fit(network, train_loader, val_loader, test_loader, epochs, lr):
    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    f1 = F1()

    train_losses, val_losses, scores = [], [], []
    val_losses.append(evaluate(network, val_loader, loss_fn))

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = update(network, train_loader, loss_fn, optimizer)
        train_losses += train_loss
        val_loss = evaluate(network, val_loader, loss_fn)
        val_losses.append(val_loss)
        
        score = evaluate(network, val_loader, f1)
        if not scores or score > max(scores):
            best_model = deepcopy(network)
        scores.append(score)

    test_score = evaluate(network, test_loader, f1)
    print(f'Final f1 on test dataset: {round(test_score * 100, 2)}')

    return train_losses, val_losses, scores, test_score


path = os.path.join(".", "dataset", "mnist")
os.makedirs(path, exist_ok=True)

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train = datasets.MNIST(path, download=True, train=True, transform=transform)
test = datasets.MNIST(path, download=True, train=False, transform=transform)

epochs = 20
lr = 1e-3
batch_size = 128
num_workers = 2
p_drop = 0.05

range_ = np.random.default_rng(seed=42)
val_inds = range_.choice(np.arange(len(train)), size=len(train) // 3, replace=False)
train_inds = np.delete(np.arange(len(train)), val_inds)

trainloader = DataLoader(Subset(train, indices=train_inds),
                         batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
valloader = DataLoader(Subset(train, indices=val_inds),
                       batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
testloader = DataLoader(test, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)

network = MLP(in_features=784, out_features=10, p_drop=p_drop, self_normalized=False).to(device)
r_train_losses, r_val_losses, r_scores, r_test_score = fit(network,
                                                           trainloader,
                                                           valloader,
                                                           testloader,
                                                           epochs,
                                                           lr)

network = MLP(in_features=784, out_features=10, p_drop=p_drop, self_normalized=True).to(device)
train_losses, val_losses, scores, test_score = fit(network,
                                                   trainloader,
                                                   valloader,
                                                   testloader,
                                                   epochs,
                                                   lr)

plt.figure()
plt.plot(np.asarray(range(len(r_train_losses))) / len(trainloader), r_train_losses, alpha=0.4, label="ReLU train loss")
plt.plot(np.asarray(range(len(train_losses))) / len(trainloader), train_losses, alpha=0.4, label="SELU train loss")
plt.plot(range(len(r_val_losses)), r_val_losses, color="C0", label="ReLU val loss")
plt.plot(range(len(val_losses)), val_losses, color="C1", label="SELU val loss")
plt.title("Train and Validation loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("CELoss")
plt.yscale("log")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, len(r_scores) + 1), [a * 100 for a in r_scores], label="ReLU val accuracy")
plt.hlines(y=r_test_score * 100, xmin=0, xmax=len(r_scores), colors="C0", linestyles="dashed", label="ReLU test accuracy")
plt.plot(range(1, len(scores) + 1), [a * 100 for a in scores], label="SELU val accuracy")
plt.hlines(y=test_score * 100, xmin=0, xmax=len(scores), colors="C1", linestyles="dashed", label="SELU test accuracy")
plt.title("Validation F1")
plt.xlabel("Epochs")
plt.ylabel("F1")

plt.legend()
plt.show()
