# This module plots supports from the FE problem
# It returns the Xnew and dsup variables for use by plotloads

import numpy as np
import matplotlib.pyplot as plt

def plotsupports(X, D, neqn, bound):
    # Finding the size of the structure and the deformed (new) coordinates
    Dx = np.arange(0, neqn, 2)
    Dy = np.arange(1, neqn, 2)
    Xnew = np.zeros_like(X)
    Xnew[:, 0] = X[:, 0] + D[Dx,0]
    Xnew[:, 1] = X[:, 1] + D[Dy,0]

    maxX, minX = np.max(Xnew[:, 0]), np.min(Xnew[:, 0])
    maxY, minY = np.max(Xnew[:, 1]), np.min(Xnew[:, 1])
    sizeX, sizeY = maxX - minX, maxY - minY

    # if size is 1 in the X-dimension used for the supports "dsup" are 0.015 
    dsup = sizeX * 0.015

    # Plotting the supports
    for b in range(bound.shape[0]):
        XX = Xnew[int(bound[b, 0])-1, 0]
        YY = Xnew[int(bound[b, 0])-1, 1]
        dsup = abs(dsup)

        if bound[b, 1] == 1:  # hold in X-direction
            if XX == maxX: 
                dsup = -dsup
            plt.plot(XX, YY, 'k.', linewidth=1.5)
            plt.plot([XX, XX-2*dsup], [YY, YY+2*dsup], 'k', linewidth=1)
            plt.plot([XX-2*dsup, XX-2*dsup], [YY-2*dsup, YY+2*dsup], 'k', linewidth=1)
            plt.plot([XX-2*dsup, XX], [YY-2*dsup, YY], 'k', linewidth=1)
            plt.plot(XX-3*dsup, YY+dsup, 'ko', mfc='none', linewidth=0.7)
            plt.plot(XX-3*dsup, YY-dsup, 'ko', mfc='none', linewidth=0.7)
            plt.plot([XX-4*dsup, XX-4*dsup], [YY-2*dsup, YY+2*dsup], 'k', linewidth=1)
            plt.plot([XX-4*dsup, XX-5*dsup], [YY+2*dsup, YY+1.005*dsup], 'k', linewidth=1)
            plt.plot([XX-4*dsup, XX-5*dsup], [YY+1*dsup, YY+0.004*dsup/0.025], 'k', linewidth=1)
            plt.plot([XX-4*dsup, XX-5*dsup], [YY, YY-0.995*dsup], 'k', linewidth=1)
            plt.plot([XX-4*dsup, XX-5*dsup], [YY-1*dsup, YY-1.990*dsup], 'k', linewidth=1)
            plt.plot([XX-4*dsup, XX-5*dsup], [YY-2*dsup, YY-2.985*dsup], 'k', linewidth=1)
        elif bound[b, 1] == 2:  # hold in the y-direction
            if YY == maxY:  
                dsup = -dsup
            plt.plot(XX, YY, 'k.', linewidth=1.5)
            plt.plot([XX, XX-2*dsup], [YY, YY-2*dsup], 'k', linewidth=1)
            plt.plot([XX-2*dsup, XX+2*dsup], [YY-2*dsup, YY-2*dsup], 'k', linewidth=1)
            plt.plot([XX+2*dsup, XX], [YY-2*dsup, YY], 'k', linewidth=1)
            plt.plot(XX-dsup, YY-3*dsup, 'ko', mfc='none', linewidth=0.7)
            plt.plot(XX+dsup, YY-3*dsup, 'ko', mfc='none', linewidth=0.7)
            plt.plot([XX-2*dsup, XX+2*dsup], [YY-4*dsup, YY-4*dsup], 'k', linewidth=1)
            plt.plot([XX+2*dsup, XX+1.01*dsup], [YY-4*dsup, YY-5*dsup], 'k', linewidth=1)
            plt.plot([XX+1*dsup, XX+0.004*dsup/0.025], [YY-4*dsup, YY-5*dsup], 'k', linewidth=1)
            plt.plot([XX, XX-0.995*dsup], [YY-4*dsup, YY-5*dsup], 'k', linewidth=1)
            plt.plot([XX-1*dsup, XX-1.990*dsup], [YY-4*dsup, YY-5*dsup], 'k', linewidth=1)
            plt.plot([XX-2*dsup, XX-2.985*dsup], [YY-4*dsup, YY-5*dsup], 'k', linewidth=1)

    return Xnew, dsup