import numpy as np
class r:
    def __init__(self, nbData, model):
        self.Data = np.zeros((model.nbVar, nbData))
        self.Mu = np.ndarray(shape=(model.nbVar, model.nbStates, nbData))
        self.Sigma = np.ndarray(shape=(model.nbVar, model.nbVar, model.nbStates, nbData))
        self.p = None
        self.H = np.ndarray(shape=(model.nbStates, nbData))
