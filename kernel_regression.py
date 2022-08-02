import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


class KernelRegression:
    '''
        Nadaraya-Watson kernel regression with heuristic brute-force
        window width selection
    '''
    def __init__(self, kernel='rbf', gamma=None):
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y):
        self.x = X
        self.y = y
        
        try:
            _ = len(self.gamma)
            self.gamma = self.find_gamma(self.gamma)
        except:
            pass

    def find_gamma(self, gammas):
        mse = np.empty_like(gammas)
        for i, gamma in enumerate(gammas):
            K = pairwise_kernels(self.x, self.x, metric=self.kernel, gamma=gamma)
            np.fill_diagonal(K, 0)
            Ky = K * self.y[:, np.newaxis]
            y_pred = Ky.sum(axis=0) / K.sum(axis=0)
            mse[i] = np.square(y_pred - self.y).mean()
        return gammas[np.nanargmin(mse)]

    def predict(self, X):
        K = pairwise_kernels(self.x, X, metric=self.kernel, gamma=self.gamma)
        Ky = K * self.y[:, np.newaxis]
        return Ky.sum(axis=0) / K.sum(axis=0)
