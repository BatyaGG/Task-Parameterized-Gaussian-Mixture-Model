# Task-Parameterized-Gaussian-Mixture-Model

![alt text](https://raw.githubusercontent.com/BatyaGG/Task-Parameterized-Gaussian-Mixture-Model/master/TPGMM_figure.PNG)

Python implementation of Task-Parameterized Gaussian Mixture Model(TPGMM) and Regression algorithms with example and data in txt format. TPGMM is Gaussian Mixture Model algorithm which is parameterized on reference frames locations and orientations. It adapts regression trajectories based on the parameters - positions and orientations of the frames. Any object or point in cartesian space is able to be a reference frame. Current approach uses k-means clustering to initialize gaussian parameters and iterative Expectation-Maximization (EM) algorithm to bring them closer to the truth. After TPGMM is fitted, the model together with new frame parameters are applied to gaussian regression to retrieve output features by time input. Please take a look to the demo video of application of TPGMM and GMR to train/generate NAO robot right arm trajectory.

<p align="center"> 
<a href="https://www.youtube.com/watch?v=yENSZpewsRY" target="_blank"><img src="https://github.com/BatyaGG/Task-Parameterized-Gaussian-Mixture-Model/blob/master/youtube.png" 
width="70%"" border="10" /></a> 
<br> 
<i>The demo video</i> 
</p> 

### Associated paper:
Alizadeh, T., & **Saduanov, B.** (2017, November). Robot programming by demonstration of multiple tasks within a common environment. In Multisensor Fusion and Integration for Intelligent Systems (MFI), 2017 IEEE International Conference on (pp. 608-613). IEEE.

All math, concepts and data are referred from the research publication and MATLAB implementation both by professor Sylvain Calinon (http://calinon.ch):

Calinon, S. (2016)
A Tutorial on Task-Parameterized Movement Learning and Retrieval
Intelligent Service Robotics (Springer), 9:1, 1-29.

### MATLAB implementation:
http://calinon.ch/download/task-parameterized-GMM.zip

Thanks to XD-DENG for his matrix square root implementation (to avoid usage of scipy): https://github.com/XD-DENG/sqrt-matrix

# Installation

Clone or download the project

Install following packages: numpy <1.11.3>, matplotlib <1.5.3>

Other versions of the packages were not tested, but higher versions are welcome. Report me to b.saduanov@gmail.com if you have any problems.

# Usage

To use TPGMM algorithm you need to have your datas in proper format:

![alt text](https://raw.githubusercontent.com/BatyaGG/Task-Parameterized-Gaussian-Mixture-Model/master/data.JPG)

where s is a sample class and p is a parameters class. List of samples should be initialized and each sample object should have raw data recorded in reference to general frame in NxM format, number of Data points which is equals to M, and matrix of parameter objects in LxM format, where N is a number of variables, M is a number of data points and L is a number of frames in a workspace. Each column in parameters matrix corresponds to same column in Data matrix, i.e. parameter points and data points were recorded simultaneously and each column of them corresponds to the same time moment. GAMMA and GAMMA0 fields should not be initialized by user and they will be filled and used in future calculations by algorithm. So in general user have to create parameters matrix and raw data matrix to initialize sample.

Parameter class have A matrix, b matrix, A inverse matrix and number of states fields. Each parameter object corresponds to specific time step and specific frame of reference. Rows of parameter object matrix defines frames and columns defines time steps. For example, at first time step, first frame was at (2,3) coordinate from general origin and had 45 degrees of rotation about general origin: so user have to create parameter object and put it in first row of parameter matrix since it is first frame and Nth column where N defines recording moment point, b matrix field of that object should be ```np.array([[0, 2, 3]]).T column vector```, A matrix field of that object should be ```np.array([[1, 0, 0],[0, 0.7, -0.7],[0, 0.7, 0.7]])``` [(rotation matrix)](https://en.wikipedia.org/wiki/Rotation_matrix). Please, note that for time dimension we put 0 in b vector and 1 followed by 0s in first row of A matrix always, because we have no dependency of trajectory on time.

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
After creating list of samples which we called ```slist``` we can fit TPGMM algorithm on it. First of all initiate TPGMM_GMR object by writing:
```
from TPGMM_GMR import TPGMM_GMR
TPGMMGMR = TPGMM_GMR(nbStates, nbFrames, nbVar)
```
Then train algorithm by calling method 'fit':
```
TPGMMGMR.fit(slist)
```
After model is learned, we can reproduce the trajectory for a new sets of parameters. For that purpose, new parameters should be recorded and created, however in this case script randomly deform original parameters matrix. To reproduce trajectory for some new task parameters, TPGMM_GMR's method 'reproduce' is used. This method have 2 parameters which are: task-parameter matrix and current position of trajectory or starting point. The method return reproduction object which have information about frames, trajectory and gaussian states.
```
reproduction = TPGMMGMR.reproduce(newTaskParameterMatrix, slist[0].Data[1:2, 0])
```
This line of code will create new trajectory based on newTaskParameterMatrix and will consider first sample's first data point as starting point.

To plot reproduction trajectory together with its task parameters and gaussian states please use TPGMM_GMR's 'plotReproduction' method.
```
fig = plt.figure()
ax = fig.add_subplot(111)
TPGMMGMR.plotReproduction(reproduction, 1, 2, ax, showGaussians=True)
```
As can be noted, Gaussians can be hided. It is useful when big amount of gaussian states is used for accuracy reason.

# Contribution
I appreciate any contribution attempts to this project. One way to contribute is to test the algorithm for several input dimensions. In general, it should work, however I did not test it yet. For this purpose, useful high dimensional data should be generated and methods have to be slightly modified. If you have any ideas or want to contribute contact me to b.saduanov@gmail.com.
