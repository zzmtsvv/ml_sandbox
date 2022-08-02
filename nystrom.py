from scipy import linalg
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


class Nystrom:
    def __init__(self, gamma, kernel='rbf', c=500, k=200, seed=42):
        self.gamma = gamma
        self.c = c
        self.k = k
        self.rand = np.random.RandomState(seed)
        self.kernel = kernel
    
    def fit(self, X_train):
        self.train = X_train
        self.n_samples = X_train.shape[0]
        self.idx = self.rand.choice(self.n_samples, self.c)
        
        self.train_idx = X_train[self.idx, :]

        self.W = pairwise_kernels(self.train_idx, self.train_idx, metric=self.kernel, gamma=self.gamma)

        u, s, vt = linalg.svd(self.W, full_matrices=False)
        self.u = u[:, :self.k]
        self.s = s[:self.k]
        self.vt = vt[:self.k, :]

        self.M = np.dot(self.u, np.diag(1 / np.sqrt(self.s)))
    
    def transform(self, X_test):
        C_train = pairwise_kernels(self.train, self.train_idx, metric=self.kernel, gamma=self.gamma)
        C_test = pairwise_kernels(X_test, self.train_idx, metric=self.kernel, gamma=self.gamma)

        X_new_train = np.dot(C_train, self.M)
        X_new_test = np.dot(C_test, self.M)

        return X_new_train, X_new_test
    
    def fit_transform(self, X_train, X_test):
        self.fit(X_train)
        new_train, new_test = self.transform(X_test)
        return new_train, new_test
