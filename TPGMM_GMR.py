from modelClass import model
from init_proposedPGMM_timeBased import init_proposedPGMM_timeBased
from EM_tensorGMM import EM_tensorGMM
from reproduction_DSGMR import reproduction_DSGMR
from plotGMM import plotGMM
import numpy as np

class TPGMM_GMR(object):
    def __init__(self, nbStates, nbFrames, nbVar):
        self.model = model(nbStates, nbFrames, nbVar, None, None, None, None, None)

    def fit(self, s):
        self.s = s
        self.model = init_proposedPGMM_timeBased(s, self.model)
        self.model = EM_tensorGMM(s, self.model)

    def reproduce(self, p, currentPosition):
        return reproduction_DSGMR(self.s[0].Data[0,:], self.model, p, currentPosition)

    def plotReproduction(self, r, xaxis, yaxis, ax, showGaussians = True, lw = 7):
        for m in range(r.p.shape[0]):
            ax.plot([r.p[m, 0].b[xaxis, 0], r.p[m, 0].b[xaxis, 0] + r.p[m, 0].A[xaxis, yaxis]],
                     [r.p[m, 0].b[yaxis, 0], r.p[m, 0].b[yaxis, 0] + r.p[m, 0].A[yaxis, yaxis]],
                     lw=lw, color=[0, 1, m])
            ax.plot(r.p[m, 0].b[xaxis, 0], r.p[m, 0].b[yaxis, 0], ms=30, marker='.', color=[0, 1, m])
        ax.plot(r.Data[xaxis, 0], r.Data[yaxis, 0], marker='.', ms=15)
        ax.plot(r.Data[xaxis, :], r.Data[yaxis, :])
        if showGaussians:
            plotGMM(r.Mu[np.ix_([xaxis, yaxis], range(r.Mu.shape[1]), [0])],
                    r.Sigma[np.ix_([xaxis, yaxis], [xaxis, yaxis], range(r.Mu.shape[1]), [0])], [0.5, 0.5, 0.5], 1, ax)

