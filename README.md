# Task-Parameterized-Gaussian-Mixture-Model

Python implementation of Task-Parameterized Gaussian Mixture Model(TPGMM) and Regression algorithms with example and data in txt format. TPGMM is Gaussian Mixture Model algorithm which is parameterized on reference frames locations and orientations. It adapts regression trajectories based on the parameters - positions and orientations of the frames. Any object or point in cartesian space is able to be a reference frame. Current approach uses k-means clustering to initialize gaussian parameters and iterative Expectation-Maximization (EM) algorithm to bring them closer to the truth. After TPGMM is fitted, the model together with new frame parameters are applied to gaussian regression to retrieve output features by time input.

All math and concepts are referred from the research publication and MATLAB implementation both by professor Sylvain Calinon (http://calinon.ch):

Calinon, S. (2016)
A Tutorial on Task-Parameterized Movement Learning and Retrieval
Intelligent Service Robotics (Springer), 9:1, 1-29.

MATLAB implementation: http://calinon.ch/download/task-parameterized-GMM.zip

Thanks to XD-DENG for his matrix square root implementation (to avoid usage of scipy): https://github.com/XD-DENG/sqrt-matrix

# Installation

Clone or download the project

Install following packages: numpy <1.11.3>, matplotlib <1.5.3>

Other versions of the packages were not tested, but higher versions are welcome. Report me to b.saduanov@gmail.com if you have any problems.

# Usage

To use TPGMM algorithm you need to have your datas in proper format:

![alt text](https://raw.githubusercontent.com/BatyaGG/Task-Parameterized-Gaussian-Mixture-Model/master/data.JPG)

where s is a sample class and p is a parameters class. List of samples should be initialized and each sample object should have raw data recorded in reference to general frame in NxM format, number of Data points which is equals to M, and matrix of parameter objects in LxM format, where N is a number of variables, M is a number of data points and L is a number of frames in a workspace. Each column in parameters matrix corresponds to same column in Data matrix, i.e. parameter points and data points were recorded simultaneously and each column of them corresponds to the same time moment. GAMMA and GAMMA0 fields should not be initialized by user and they will be filled and used in future calculations by algorithm. So in general user have to create parameters matrix and raw data matrix to initialize sample.

Parameter class have A matrix, b matrix, A inverse matrix and number of states fields. Each parameter object corresponds to specific time step and specific frame of reference. Rows of parameter object matrix defines frames and columns defines time steps. For example, at first time step, first frame was at (2,3) coordinate and had 45 degrees of rotation about origin: so user have to create parameter object and put it in first row of parameter matrix since it is first frame, b matrix field of that object should be ```np.array([[0, 2, 3]]).T column vector```, A matrix field of that object should be ```np.array([[1, 0, 0],[0, 0.7, -0.7],[0, 0.7, 0.7]])```

For particular case, this is how I create sample list using txt files in folder. I did not save the sample list in pickle or mat file to show users how it should be done. You can open txt files to understand better data format and please note that A matrices are 3x3 matrices, however in txt files I saved them concatenating all A matrices horizontally, so each 4th column is a new A matrix of next time step. b matrixes are 3x1 vectors and I also saved them concatenating horizontally, so each new column of b matrix corresponds to new time step b vector. 
```
from sClass import s
from pClass import p
import numpy as np

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
```
