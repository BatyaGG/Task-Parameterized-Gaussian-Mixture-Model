import numpy as np
class s:
    def __init__(self, p, Data, nbData, nbStates):
        self.p = p
        self.Data = Data
        self.nbData = nbData
        self.GAMMA0 = np.zeros(shape=(nbStates, self.nbData))
        self.GAMMA = None