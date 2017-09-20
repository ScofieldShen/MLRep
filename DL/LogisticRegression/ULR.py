import numpy as np
from astropy.units import *
import math
def computeCorelation(X,Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXBar = X[i] - xBar
        diffYBar = Y[i] - yBar
        SSR += diffXBar*diffYBar
        varX += diffXBar**2
        varY += diffYBar**2
    SST = math.sqrt(varX*varY)
    return SSR/SST


if __name__ == '__main__':
    testX = [1, 3, 8, 7, 9]
    testY = [10, 12, 24, 21, 34]
    print(computeCorelation(testX, testY))