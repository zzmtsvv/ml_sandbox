import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


class IsoReg:
    '''
    isotonic regression, for details check
    
        min sum w[i] (y[i] - y_[i]) ** 2
        subject to y_min = y_[1] <= y_[2] ... <= y_[n] = y_max

    https://en.wikipedia.org/wiki/Isotonic_regression
    '''
    def __init__(self, random_seed=42, callback=None):
        np.random.RandomState(random_seed)
        self.callback = callback
    
    def fit_transform(self, y, weight=None, y_min=None, y_max=None):
        if weight is None:
            weight = np.ones(len(y), dtype=y.dtype)
        
        if y_min is not None or y_max is not None:
            self.y = np.copy(y)
            self.weight = np.copy(weight)
            C = np.dot(self.weight, self.y * self.y)  # upper bound on MSE cost function
            if y_min is not None:
                self.y[0] = y_min
                self.weight[0] = C
            if y_max is not None:
                self.y[-1] = y_max
                self.weight[-1] = C
        
        self.active_set = [(self.weight[i] * self.y, self.weight[i], [i, ]) for i in range(len(self.y))]

        current, counter = 0, 0

        while current < len(self.active_set) - 1:
            value0, value1, value2 = 0, 0, np.inf
            weight0, weight1, weight2 = 0, 0, 0

            while value0 * weight1 <= value1 * weight0 and current < len(self.active_set) - 1:
                value0, weight0, idx0 = self.active_set[current]
                value1, weight1, idx1 = self.active_set[current + 1]
                if value0 * weight1 <= value1 * weight0:
                    current += 1
                
                if self.callback is not None:
                    self.callback(y, self.active_set, counter, idx1)
                    counter += 1
                
            if current == len(self.active_set) - 1:
                break

            value0, weight0, idx0 = self.active_set.pop(current)
            value1, weight1, idx1 = self.active_set.pop(current)
            self.active_set.insert(current, (value0 + value1, weight0 + weight1, idx0 + idx1))

            while value2 * weight0 > value0 * weight2 and current > 0:
                value0, weight0, idx0 = self.active_set[current]
                value2, weight2, idx2 = self.active_set[current - 1]

                if weight0 * value2 >= weight2 * value0:
                    del self.active_set[current]

                    self.active_set[current - 1] = (value0 + value2, weight0 + weight2, idx0 + idx2)
                    current -= 1
        self.solution = np.empty(len(self.y))

        if self.callback is not None:
            self.callback(self.y, self.active_set, counter + 1, idx1)
            self.callback(self.y, self.active_set, counter + 2, idx1)
        
        for value, weight, idx in self.active_set:
            self.solution[idx] = value / weight
        
        return self.solution


class LogReg(nn.Module):
    @property
    def device(self):
        for p in self.parameters():
            return p.device()
    
    def __init__(self, input_dim, output_dim):
        super(LogReg, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class PlattScaling:
    '''
    Additionally, the logistic model works best if the calibration error is symmetrical,
    meaning the classifier output for each binary class is normally distributed with the same variance.
    This can be a problem for highly imbalanced classification problems, where outputs do not have equal variance.
    In general this method is most effective when the un-calibrated model is under-confident and has similar
    calibration errors for both high and low outputs.
    '''
    def __init__(self, uncalibrated_model, optimizer=Adam, criterion=nn.BCELoss):
        self.uncalibrated_model = uncalibrated_model
        self.criterion = criterion()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logreg = LogReg(input_dim=1, output_dim=1)
        self.logreg.to(self.device)
        self.optim = optimizer(self.logreg.parameters())

    def fit(self, X, y, epochs=20):
        '''
        predictions of the model must be -1 or 1 and take shape of (n_samples, 1)
        '''
        f = self.uncalibrated_model.predict(X)
        y_hat = self.create_new_targets(y)

        f = torch.Tensor(f, device=self.device)
        y_hat = torch.Tensor(y_hat, device=self.device)

        # training on the whole training set

        for epoch in tqdm(range(1, epochs + 1), desc='Training epochs'):
            self.optim.zero_grad()
            outputs = self.logreg(f)
            
            loss = self.criterion(torch.squeeze(outputs), torch.squeeze(y_hat))
            loss.backward()

            self.optim.step()
    
    def predict_prob(self, X):
        f = self.uncalibrated_model(X)
        
        with torch.no_grad():
            f = torch.Tensor(f, device=self.device)
            prob = self.logreg(f)

        out = prob.cpu().detach().numpy()
        return out

    
    def create_new_targets(self, y):
        n_positive = 0 # number of positive samples
        n_negative = 0 # number of negative samples

        # костыль
        y = np.array(y).flatten()
        for label in y:
            if label > 0:
                n_positive += 1
        n_negative = len(y) - n_positive

        # according to the bayes rule
        proba_pos = (n_positive + 1) / (n_positive + 2)
        proba_neg = 1 / (n_negative + 2)

        # new tagets (probabilistic)
        t = [proba_pos if y[i] > 0 else proba_neg for i in range(len(y))]
        t = np.array(t).T
        return t
