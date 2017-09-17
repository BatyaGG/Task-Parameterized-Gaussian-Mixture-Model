import numpy as np
from sClass import s
from pClass import p
from matplotlib import pyplot as plt
from modelClass import model
from init_proposedPGMM_timeBased import init_proposedPGMM_timeBased
from EM_tensorGMM import EM_tensorGMM
from reproduction_DSGMR import reproduction_DSGMR
from plotGMM import plotGMM

nbSamples = 4
nbVar = 3
nbFrames = 2
nbStates = 10
nbData = 200

# Preparing the samples----------------------------------------------------------------------------------------------- #
slist = []
for i in range(nbSamples):
    pmat = np.empty(shape=(nbFrames, nbData), dtype=object)
    tempData = np.loadtxt('sample' + str(i + 1) + '_Data.txt', delimiter=',')
    for j in range(nbFrames):
        tempA = np.loadtxt('sample' + str(i + 1) + '_frame' + str(j + 1) + '_A.txt', delimiter=',')
        tempB = np.loadtxt('sample' + str(i + 1) + '_frame' + str(j + 1) + '_b.txt', delimiter=',')
        for k in range(nbData):
            pmat[j, k] = p(tempA[:, 3*k : 3*k + 3], tempB[:, k].reshape(len(tempB[:, k]), 1),
                           np.linalg.inv(tempA[:, 3*k : 3*k + 3]), nbStates)
    slist.append(s(pmat, tempData, tempData.shape[1], nbStates))
# -------------------------------------------------------------------------------------------------------------------- #


# Learning the model-------------------------------------------------------------------------------------------------- #
model = model(nbStates, nbFrames, nbVar, None, None, None, None, None)
model = init_proposedPGMM_timeBased(slist, model)
model, tensor = EM_tensorGMM(slist, model)
# -------------------------------------------------------------------------------------------------------------------- #

# Reproduction for parameters used in demonstration------------------------------------------------------------------- #
rlist = []
for n in range(nbSamples):
    rlist.append(reproduction_DSGMR(slist[0].Data[0,:], model, slist[n].p, slist[n].Data[1:3,0]))
# -------------------------------------------------------------------------------------------------------------------- #

# Plotting------------------------------------------------------------------------------------------------------------ #
xaxis = 1
yaxis = 2
xlim = [-10, 10]
ylim = [-10, 10]

# Demos--------------------------------------------------------------------------------------------------------------- #
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))

for n in range(nbSamples):
    for m in range(nbFrames):
        ax1.plot([slist[n].p[m,0].b[xaxis,0], slist[n].p[m,0].b[xaxis,0] + slist[n].p[m,0].A[xaxis,yaxis]], [slist[n].p[m,0].b[yaxis,0], slist[n].p[m,0].b[yaxis,0] + slist[n].p[m,0].A[yaxis,yaxis]], lw = 7, color = [0,1,m])
        ax1.plot(slist[n].p[m,0].b[xaxis,0], slist[n].p[m,0].b[yaxis,0], ms = 30, marker = '.', color = [0,1,m])
    ax1.plot(slist[n].Data[xaxis,0], slist[n].Data[yaxis,0], marker = '.', ms = 15)
    ax1.plot(slist[n].Data[xaxis,:], slist[n].Data[yaxis,:])
# -------------------------------------------------------------------------------------------------------------------- #

# Reproductions with training parameters------------------------------------------------------------------------------ #
ax2 = fig.add_subplot(132)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))

for n in range(nbSamples):
    for m in range(nbFrames):
        ax2.plot([rlist[n].p[m,0].b[xaxis,0], rlist[n].p[m,0].b[xaxis,0] + rlist[n].p[m,0].A[xaxis,yaxis]], [rlist[n].p[m,0].b[yaxis,0], rlist[n].p[m,0].b[yaxis,0] + rlist[n].p[m,0].A[yaxis,yaxis]], lw = 7, color = [0,1,m])
        ax2.plot(rlist[n].p[m,0].b[xaxis,0], rlist[n].p[m,0].b[yaxis,0], ms = 30, marker = '.', color = [0,1,m])
    ax2.plot(rlist[n].Data[xaxis, 0], rlist[n].Data[yaxis, 0], marker='.', ms=15)
    ax2.plot(rlist[n].Data[xaxis, :], rlist[n].Data[yaxis, :])
    # plotGMM(rlist[n].Mu[np.ix_([xaxis,yaxis], range(nbStates), [1])], rlist[n].Sigma[np.ix_([xaxis, yaxis], [xaxis, yaxis], range(nbStates), [1])], [0, 0.8, 0], 1, ax2)


# -------------------------------------------------------------------------------------------------------------------- #

nclus = nbStates
frameIndex = 0
rows = np.array([xaxis, yaxis])
cols = np.arange(0, nclus, 1)
plotGMM(model.ref[frameIndex].ZMu[np.ix_(rows, cols)], model.ref[frameIndex].ZSigma[np.ix_(rows, rows, cols)],
    [0, 0.8, 0], 1, ax2)

plt.show()