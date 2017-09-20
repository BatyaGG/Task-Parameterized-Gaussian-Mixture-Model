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

To use TPGMM algorithm you need to have your datas in proper format. 
