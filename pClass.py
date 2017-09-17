import numpy as np
class p:
    def __init__(self, A, b, invA, nbStates):
        self.A = A
        self.b = b
        self.invA = invA
        self.Mu = np.zeros(shape=(len(b),nbStates))
        self.Sigma = np.zeros(shape=(len(b),len(b),nbStates))