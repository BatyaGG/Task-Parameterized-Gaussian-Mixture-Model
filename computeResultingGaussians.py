import numpy as np
from numpy import linalg
def computeResultingGaussians(model, pp):
    from prodResClass import prodRes
    prodResList = []
    for t in range (0,np.shape(pp)[1]):
        # MuTmp = np.ndarray(shape=(model.nbVar, 0))
        for i in range (0, model.nbStates):
            for m in range (0, model.nbFrames):
                # print(np.shape(np.dot(pp[m, t].A, np.reshape(model.ref[m].ZMu[:, i], newshape=(model.nbVar, 1))) + pp[m, t].b))
                # np.dot(pp[m, t].A, np.reshape(model.ref[m].ZMu[:, i], newshape=(model.nbVar, 1))) + pp[m, t].b
                pp[m, t].Mu[:,i] = np.reshape(np.dot(pp[m, t].A, np.reshape(model.ref[m].ZMu[:, i], newshape=(model.nbVar, 1))) + pp[m, t].b, (model.nbVar,))
                a = np.dot(pp[m, t].A, np.reshape(model.ref[m].ZSigma[:, :, i], newshape=(model.nbVar, model.nbVar)))
                pp[m, t].Sigma[:,:,i] = np.dot(a, pp[m, t].invA) + np.identity(model.nbVar)*0.00000001
    for t in range (0, np.shape(pp)[1]):
        prodResList.append(prodRes(model.nbVar))
        for i in range (0, model.nbStates):
            SigmaTmp = np.zeros((model.nbVar, model.nbVar))
            MuTmp = np.zeros((model.nbVar,1))
            for m in range (0, model.nbFrames):
                SigmaTmp = SigmaTmp + np.linalg.inv(pp[m, t].Sigma[:, :, i])
                MuTmp = MuTmp + np.dot(np.linalg.inv(pp[m, t].Sigma[:,:,i]), np.reshape(pp[m, t].Mu[:,i], newshape=(model.nbVar, 1)))
            prodResList[t].invSigma = np.dstack((prodResList[t].invSigma, SigmaTmp))
            prodResList[t].Sigma = np.dstack((prodResList[t].Sigma, np.linalg.inv(SigmaTmp)))
            prodResList[t].detSigma = np.hstack((prodResList[t].detSigma, np.linalg.det(prodResList[t].Sigma[:,:,i])))
            prodResList[t].Mu = np.hstack((prodResList[t].Mu, np.dot(prodResList[t].Sigma[:,:,i], MuTmp)))
            # print np.dot(prodResList[t].Sigma[:,:,i], MuTmp)
    return prodResList, pp