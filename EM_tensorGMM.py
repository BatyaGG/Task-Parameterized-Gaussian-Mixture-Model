import numpy as np
import matplotlib.pyplot as plt
def EM_tensorGMM(s, model):
    nbMinSteps = 5
    nbMaxSteps = 100
    maxDiffLL = 0.00001
    nbSamples = len(s)
    nbDataTotal = 0
    for i in range(0, nbSamples):
        nbDataTotal = nbDataTotal + s[i].nbData
    tensorData = np.zeros((model.nbVar, model.nbFrames, nbDataTotal))
    DataIndStart = 0
    DataIndEnd = 0
    for n in range(0, nbSamples):
        S = s[n]
        DataIndEnd = DataIndEnd + S.nbData
        for m in range(0, model.nbFrames):
            tensorData[np.ix_(range(0, model.nbVar), range(m, m + 1), range(DataIndStart, DataIndEnd))] = np.reshape(np.linalg.solve(S.p[m, 0].A, S.Data) - np.tile(np.reshape(S.p[m,0].b, (model.nbVar, 1)), (1, S.nbData)), (model.nbVar, 1, (DataIndEnd - DataIndStart)))
        DataIndStart = DataIndEnd



    nbData = np.shape(tensorData)[2]
    diagRegularizationFactor = 0.00000001
    LL = []
    for nbIter in range(0, nbMaxSteps):
        L, GAMMA, GAMMA0 = computeGamma(tensorData, model)
        GAMMA2 = GAMMA / np.tile(np.reshape(np.sum(GAMMA,axis=1), (np.shape(GAMMA)[0], 1)), (1, nbData))

        for i in range(0, model.nbStates):
            model.Priors[i] = np.sum(np.sum(GAMMA[i, :])) / nbData

            for m in range(0, model.nbFrames):
                DataMat = tensorData[:,m,:]
                model.ref[m].ZMu[:,i] = np.reshape(np.dot(DataMat, np.reshape(GAMMA2[i,:], (1, np.shape(GAMMA2)[1])).T), (model.nbVar,))
                DataTmp = DataMat - np.tile(np.reshape(model.ref[m].ZMu[:,i], (np.shape(model.ref[m].ZMu)[0], 1)), (1, nbData))
                a = np.dot(DataTmp, np.diag(GAMMA2[i, :]))
                model.ref[m].ZSigma[:, :, i] = np.dot(a, DataTmp.T) + np.eye(np.shape(DataTmp)[0])*diagRegularizationFactor
        LL.append(sum(np.log(np.sum(L, axis=0)))/np.shape(L)[1])
        if nbIter>nbMinSteps:
            if LL[nbIter] - LL[nbIter-1] < maxDiffLL or nbIter==nbMaxSteps-1:
                print 'EM converged after ' + str(nbIter) + ' iterations'
                return model, tensorData

    print 'The maximum number of ' + str(nbMaxSteps) + ' EM iterations has been reached'
    return model, tensorData

def computeGamma(Data, model):
    from gaussPDFfast import gaussPDFfast
    import sys
    realmin = sys.float_info[3]
    nbData = np.shape(Data)[2]
    Lik = np.ones((model.nbStates, nbData))
    GAMMA0 = np.zeros((model.nbStates, model.nbFrames, nbData))
    for i in range(0, model.nbStates):
        for m in range(0, model.nbFrames):
            DataMat = Data[:,m,:]
            GAMMA0[i, m, :] = gaussPDFfast(DataMat, model.ref[m].ZMu[:,i], np.linalg.inv(model.ref[m].ZSigma[:,:,i]), np.linalg.det(model.ref[m].ZSigma[:,:,i]))
            Lik[i, :] = Lik[i,:]*np.squeeze(GAMMA0[i,m,:])
        Lik[i,:] = np.dot(Lik[i,:], model.Priors[i])
    GAMMA = Lik / np.tile((np.sum(Lik, axis=0) + realmin), (np.shape(Lik)[0], 1))
    return Lik, GAMMA, GAMMA0