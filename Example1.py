import numpy as np
from sClass import s
from pClass import p
from matplotlib import pyplot as plt
from TPGMM_GMR import TPGMM_GMR
from copy import deepcopy

# Initialization of parameters and properties------------------------------------------------------------------------- #
nbSamples = 4
nbVar = 3
nbFrames = 2
nbStates = 3
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

# Creating instance of TPGMM_GMR-------------------------------------------------------------------------------------- #
TPGMMGMR = TPGMM_GMR(nbStates, nbFrames, nbVar)

# Learning the model-------------------------------------------------------------------------------------------------- #
TPGMMGMR.fit(slist)

# Reproduction for parameters used in demonstration------------------------------------------------------------------- #
rdemolist = []
for n in range(nbSamples):
    rdemolist.append(TPGMMGMR.reproduce(slist[n].p, slist[n].Data[1:3,0]))

# Reproduction with generated parameters------------------------------------------------------------------------------ #
rnewlist = []
for n in range(nbSamples):
    newP = deepcopy(slist[n].p)
    for m in range(1, nbFrames):
        bTransform = np.random.rand(nbVar, 1) + 0.5
        aTransform = np.random.rand(nbVar, nbVar) + 0.5
        for k in range(nbData):
            newP[m, k].A = newP[m, k].A * aTransform
            newP[m, k].b = newP[m, k].b * bTransform
    rnewlist.append(TPGMMGMR.reproduce(newP, slist[n].Data[1:nbVar, 0]))

# Plotting------------------------------------------------------------------------------------------------------------ #
xaxis = 1
yaxis = 2
xlim = [-1.2, 0.8]
ylim = [-1.1, 0.9]

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

# Reproductions with training parameters------------------------------------------------------------------------------ #
ax2 = fig.add_subplot(132)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
for n in range(nbSamples):
    TPGMMGMR.plotReproduction(rdemolist[n], xaxis, yaxis, ax2, showGaussians=True)

# Reproductions with new parameters----------------------------------------------------------------------------------- #
ax3 = fig.add_subplot(133)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
for n in range(nbSamples):
    TPGMMGMR.plotReproduction(rnewlist[n], xaxis, yaxis, ax3, showGaussians=True)

plt.show()