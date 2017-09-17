import numpy as np
def reproduction_DSGMR(DataIn, model, rr, currPos):
    # DataIn = np.linspace(np.amin(DataIn), np.amax(DataIn), np.shape(DataIn)[0])
    from sklearn import tree
    model.dt = 0.01
    model.kP = 150
    model.kV = 20
    DataIn = np.reshape(DataIn, newshape=(1, np.shape(DataIn)[0]))
    from rClass import r
    from computeResultingGaussians import computeResultingGaussians
    from gaussPDFfast import gaussPDFfast
    nbData = np.shape(DataIn)[1]
    iN = range(0, np.shape(DataIn)[0])
    out = range(iN[-1]+1, model.nbVar)
    nbVarOut = len(out)
    currPos = np.reshape(currPos, newshape=(nbVarOut,1))

    a = r(nbData,model)
    a.Data = np.zeros((nbVarOut+len(iN), np.shape(DataIn)[1]))
    a.Data[np.ix_(range(0,np.shape(DataIn)[0]), range(0, np.shape(DataIn)[1]))] = DataIn

    prodRes, a.p = computeResultingGaussians(model, rr)
    for t in range(0, len(prodRes)):
        a.Mu[:,:,t] = prodRes[t].Mu
        a.Sigma[:,:,:,t] = prodRes[t].Sigma
        for i in range(0,model.nbStates):
            prodRes[t].invSigmaIn = np.dstack((prodRes[t].invSigmaIn, 1/prodRes[t].Sigma[iN, iN,i][0]))
            prodRes[t].detSigmaIn = np.hstack((prodRes[t].detSigmaIn, prodRes[t].Sigma[iN, iN, i][0]))

    currVel = np.zeros(shape=(nbVarOut,1))
    y = np.zeros((len(out), nbData))
    for n in range(0, nbData):
        if len(prodRes) > 1:
            nn = n
        else:
            nn = 1
        for i in range(0, model.nbStates):
            a.H[i,n] = model.Priors[i] * gaussPDFfast(np.reshape(DataIn[:,n], newshape=(1,1)), prodRes[nn].Mu[iN, i], prodRes[nn].invSigmaIn[iN,iN,i], prodRes[nn].detSigmaIn[i])
        a.H[:,n] = a.H[:,n]/np.sum(a.H[:,n])
        # MuTmp = np.zeros((model.nbVar-len(iN), model.nbStates))
        currTar = np.zeros((nbVarOut,1))
        for i in range(0, model.nbStates):
            MuTmp = np.reshape(np.reshape(prodRes[nn].Mu[out,i], newshape=(nbVarOut, 1)) + np.reshape(prodRes[nn].Sigma[out, iN, i], newshape=(nbVarOut,len(iN))) * 1/prodRes[nn].Sigma[iN,iN,i] * (DataIn[:,n] - prodRes[nn].Mu[iN, i]), newshape=(len(out)))
            currTar = currTar + a.H[i,n] * np.reshape(MuTmp, newshape=(nbVarOut,1))
            # y[:,n] = y[:,n] + a.H[i,n]*MuTmp
        currAcc = model.kP * (currTar - currPos) - model.kV * currVel
        currVel = currVel + currAcc * model.dt
        currPos = currPos + currVel * model.dt
        a.Data[:,n] = np.reshape(np.vstack((DataIn[:,n], currPos)), newshape=(nbVarOut + np.size(iN),))
        # expData, expSigma, Mu, Sigma = process(a.Data, 5, 100)
        # a.Data = np.vstack((DataIn, y))
    return a