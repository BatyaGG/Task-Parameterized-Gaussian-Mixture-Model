import numpy as np
def init_proposedPGMM_timeBased(s, modelcur):
    from refClass import ref
    from modelClass import model
    diagRegularizationFactor = 0.0001
    nbSamples = len(s)
    DataTotalSize = 0
    for i in range(0, len(s)):
        DataTotalSize = DataTotalSize + s[i].nbData
    DataAll = np.ndarray(shape = (0,DataTotalSize))
    for i in range (0, modelcur.nbFrames):
        DataTmp = np.ndarray(shape=(np.shape(s[0].Data)[0], 0))
        for j in range (0, nbSamples):
            for k in range (0, s[j].nbData):
                # print np.shape(np.dot(s[j].p[i, k].invA, (np.reshape(s[j].Data[:, k], newshape=(np.shape(s[0].Data)[0], 1)) - np.reshape(s[j].p[i, k].b, newshape=(np.shape(s[0].Data)[0], 1)))))
                DataTmp = np.append(DataTmp, np.dot(s[j].p[i,k].invA,(np.reshape(s[j].Data[:,k], newshape = (np.shape(s[0].Data)[0],1))-np.reshape(s[j].p[i, k].b, newshape=(np.shape(s[0].Data)[0], 1)))), axis = 1)
        DataAll = np.append(DataAll, DataTmp, axis=0)
    TimingSep = np.linspace(np.amin(DataAll[0,:]), np.amax(DataAll[0,:]), num = modelcur.nbStates+1)
    Priors = []
    Mu = np.ndarray(shape=(np.shape(DataAll)[0], 1))
    Mu = np.delete(Mu, 0, axis = 1)
    Sigma = np.ndarray(shape=(np.shape(DataAll)[0], np.shape(DataAll)[0], 1))
    Sigma = np.delete(Sigma, 0, axis = 2)
    for i in range (0, modelcur.nbStates):
        idtmp = np.intersect1d(np.nonzero(DataAll[0,:] >= TimingSep[i]), np.nonzero(DataAll[0,:] < TimingSep[i+1]))
        Priors.append(len(idtmp))
        muData = DataAll[np.ix_(np.arange(0, np.shape(DataAll)[0]), idtmp)].T
        Mu = np.append(Mu, np.reshape(np.mean(muData, axis = 0), newshape = (np.shape(DataAll)[0], 1)), axis = 1)
        Sigma = np.append(Sigma, np.reshape(np.cov(muData.T) + np.identity(np.shape(DataAll)[0])*diagRegularizationFactor, newshape = (np.shape(DataAll)[0],np.shape(DataAll)[0],1)), axis = 2)
    Priors = [float(x) / sum(Priors) for x in Priors]

    reflist = []
    for i in range(0, modelcur.nbFrames):
        ZMuTmp = Mu[np.ix_(range(i*modelcur.nbVar,(i+1)*modelcur.nbVar), range(0,modelcur.nbStates))]
        ZSigmaTmp = Sigma[np.ix_(range(i*modelcur.nbVar, (i+1)*modelcur.nbVar), range(i*modelcur.nbVar, (i+1)*modelcur.nbVar), range(0, modelcur.nbStates))]
        ZSigmaTmp = ZSigmaTmp + np.tile(np.reshape(np.identity(modelcur.nbVar)*0.000001, (modelcur.nbVar, modelcur.nbVar, 1)), (1,1,modelcur.nbStates))
        reflist.append(ref(ZMuTmp, ZSigmaTmp))
    return model(modelcur.nbStates, modelcur.nbFrames, modelcur.nbVar, reflist, Priors, None, None, None)