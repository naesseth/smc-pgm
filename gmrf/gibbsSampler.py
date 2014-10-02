"""gibbsSampler.py

2D square lattice Gaussian MRF, naive Gibbs sampler.

"""
#import argparse

import numpy as np
from scipy import misc
#import matplotlib.pyplot as plt

def main():
    M = 10
    # Initialize arrays/matrices
    nx = 10
    ny = 10
    iterNr = 1000
    resultGibbs = np.zeros( (nx, ny) )
    
    # Load data
    Y = np.loadtxt('./inputData/' + str(nx) + 'x' + str(ny) + 'measurements.txt',delimiter=',')
    Y = Y.reshape( (nx, ny) )
    interactionPrec = np.loadtxt('./inputData/' + str(nx) + 'x' + str(ny) + 'intPrecision.txt',delimiter=',')
    observationPrec = np.loadtxt('./inputData/' + str(nx) + 'x' + str(ny) + 'obsPrecision.txt',delimiter=',')
    
    # Produce output
    filename = './' + str(nx) + 'x' + str(ny) + 'resultsGibbs.txt'
    f = open(filename, 'w')
    f.write('nx ny\n')
    f.write(str(nx) + ' ' + str(ny) + '\n')
    f.write('iterNr flat2Darray \n')
    f.close()
    
    # Main loop
    for iIter in np.arange(0,iterNr):
        print iIter+1
        for iRow in range(nx):
            for jCol in range(ny):
                tempAvg = Y[iRow,jCol]
                indC = ravel_multi_index((iRow, jCol), (nx, ny))
                observationVar = 1 / observationPrec[indC]
                tempVar = observationVar
                if iRow > 0:
                    indN = ravel_multi_index((iRow-1, jCol), (nx, ny))
                    interactionVar = 1 / interactionPrec[indC,indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*resultGibbs[iRow-1,jCol]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
                if iRow < nx-1:
                    indN = ravel_multi_index((iRow+1, jCol), (nx, ny))
                    interactionVar = 1 / interactionPrec[indC,indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*resultGibbs[iRow+1,jCol]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
                if jCol > 0:
                    indN = ravel_multi_index((iRow, jCol-1), (nx, ny))
                    interactionVar = 1 / interactionPrec[indC,indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*resultGibbs[iRow,jCol-1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
                if jCol< ny-1:
                    indN = ravel_multi_index((iRow, jCol+1), (nx, ny))
                    interactionVar = 1 / interactionPrec[indC,indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*resultGibbs[iRow,jCol+1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
                resultGibbs[iRow,jCol] = tempAvg + np.sqrt(tempVar) * np.random.randn()
                
        f = open(filename, 'a')
        f.write(str(iIter+1) + ' ')
        np.savetxt(f, resultGibbs.reshape( (1, nx*ny) ))
        f.close()

def ravel_multi_index(coord, shape):
    return coord[0] * shape[1] + coord[1]

if __name__ == "__main__":
    main()
