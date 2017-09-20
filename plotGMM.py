import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches

def Denman_Beavers_sqrtm(A):
    Y = A
    Z = np.eye(len(A))
    error = 1
    error_tolerance = 1.5e-8
    flag = 1
    while(error > error_tolerance):
        Y_old = Y
        Y = (Y_old + np.linalg.inv(Z))/2
        Z = (Z + np.linalg.inv(Y_old))/2
        error_matrix = abs(Y - Y_old)
        error = 0
        # detect the maximum value in the error matrix
        for i in range(len(A)):
            temp_error = max(error_matrix[i])
            if(temp_error > error):
                error = temp_error
        flag = flag + 1
    return Y

def plotGMM(Mu, Sigma, color,display_mode, ax):
    Mu = np.squeeze(Mu)
    Sigma = np.squeeze(Sigma)
    a, nbData = np.shape(Mu)
    lightcolor = np.asarray(color) + np.asarray([0.6,0.6,0.6])
    a = np.nonzero(lightcolor > 1)
    lightcolor[a] = 1

    minsx = []
    maxsx = []
    minsy = []
    maxsy = []

    if display_mode==1:
        nbDrawingSeg = 36
        t = np.linspace(-np.pi,np.pi,nbDrawingSeg)
        t = np.transpose(t)
        for j in range (0,nbData):
            stdev = Denman_Beavers_sqrtm(0.1*Sigma[:,:,j])
            X = np.dot(np.transpose([np.cos(t), np.sin(t)]), np.real(stdev))
            X = X + np.tile(np.transpose(Mu[:,j]), (nbDrawingSeg,1))

            minsx.append(min(X[:,0]))
            maxsx.append(max(X[:,0]))
            minsy.append(min(X[:,1]))
            maxsy.append(max(X[:,1]))

            verts = []
            codes = []
            for i in range (0, nbDrawingSeg+1):
                if i==0:
                    vert = (X[0,0], X[0,1])
                    code = Path.MOVETO
                elif i!=nbDrawingSeg:
                    vert = (X[i,0], X[i,1])
                    code = Path.CURVE3
                else:
                    vert = (X[0,0], X[0,1])
                    code = Path.CURVE3
                verts.append(vert)
                codes.append(code)
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor=lightcolor, edgecolor=color, lw=2, zorder = 3)
            ax.add_patch(patch)
            ax.plot(Mu[0,:], Mu[1,:], "x",color = color, zorder = 3)
        # ax.set_xlim(min(minsx),max(maxsx))
        # ax.set_ylim(min(minsy),max(maxsy))
    elif display_mode == 2:
        nbDrawingSeg = 40
        t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
        t = np.transpose(t)

        for j in range(0, nbData):
            stdev = Denman_Beavers_sqrtm(1 * Sigma[:, :, j])
            X = np.dot(np.transpose([np.cos(t), np.sin(t)]), np.real(stdev))
            X = X + np.tile(np.transpose(Mu[:, j]), (nbDrawingSeg, 1))

            minsx.append(min(X[:, 0]))
            maxsx.append(max(X[:, 0]))
            minsy.append(min(X[:, 1]))
            maxsy.append(max(X[:, 1]))

            verts = []
            codes = []
            for i in range(0, nbDrawingSeg+1):
                if i == 0:
                    vert = (X[0, 0], X[0, 1])
                    code = Path.MOVETO
                elif i != nbDrawingSeg:
                    vert = (X[i, 0], X[i, 1])
                    code = Path.CURVE3
                else:
                    vert = (X[0, 0], X[0, 1])
                    code = Path.CURVE3
                verts.append(vert)
                codes.append(code)
            path = Path(verts, codes)
            patch = patches.PathPatch(path, linestyle=None, color = lightcolor)
            ax.add_patch(patch)
        ax.plot(Mu[0, :], Mu[1, :], "-",lw = 3, color=color)
        # ax.set_xlim(min(minsx), max(maxsx))
        # ax.set_ylim(min(minsy), max(maxsy))