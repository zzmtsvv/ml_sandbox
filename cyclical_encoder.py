import numpy as np


class CyclicEncoder:
    '''
        encodes feature x to (cos(2 * pi * x / amplitude), sin(2 * pi * x / amplitude)
    '''
    def __init__(self, amplitude):
        self.amplitude = amplitude
    
    def fit(self, x):
        pass

    def transform(self, x):
        argument = 2 * np.pi * x / self.amplitude
        cos = np.cos(argument)
        sin = np.sin(argument)
        return np.vstack([cos, sin]).T
    
    def inverse_transform(self, x):
        sin = x[:, 1]
        cos = x[:, 0]
        angle = np.arctan(sin / cos) + np.pi / 2 * (1 - np.sign(cos))
        return angle * self.amplitude / (2 * np.pi)
    
    def fit_transform(self, x):
        return self.transform(x)
