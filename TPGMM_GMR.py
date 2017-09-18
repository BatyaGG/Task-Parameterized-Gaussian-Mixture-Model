from modelClass import model
from init_proposedPGMM_timeBased import init_proposedPGMM_timeBased
from EM_tensorGMM import EM_tensorGMM
from reproduction_DSGMR import reproduction_DSGMR

class TPGMM_GMR(object):
    def __init__(self, nbStates, nbFrames, nbVar):
        self.model = model(nbStates, nbFrames, nbVar, None, None, None, None, None)

    def fit(self, s):
        self.s = s
        self.model = init_proposedPGMM_timeBased(s, self.model)
        self.model = EM_tensorGMM(s, self.model)

    def reproduce(self, p, currentPosition):
        return reproduction_DSGMR(self.s[0].Data[0,:], self.model, p, currentPosition)