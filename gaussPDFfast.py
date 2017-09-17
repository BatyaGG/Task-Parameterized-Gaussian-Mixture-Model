import numpy as np
import math
import sys

def gaussPDFfast(Data, Mu, invSigma, detSigma):
    realmin = sys.float_info[3]
    nbVar, nbData = np.shape(Data)
    Data = np.transpose(Data) - np.tile(np.transpose(Mu), (nbData, 1))
    prob = np.sum(np.dot(Data, invSigma)*Data, 1)
    prob = np.exp(-0.5*prob)/np.sqrt((np.power((2*math.pi), nbVar))*(detSigma+realmin))
    return prob
