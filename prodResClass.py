import numpy as np
class prodRes:
    def __init__(self, b):
        self.invSigma = np.ndarray(shape=(b,b, 0))
        self.Sigma = np.ndarray(shape=(b, b, 0))
        self.detSigma = np.ndarray(shape=(0))
        self.Mu = np.ndarray(shape=(b, 0))
        self.invSigmaIn = np.ndarray(shape=(1,1,0))
        self.detSigmaIn = np.ndarray(shape=(0))