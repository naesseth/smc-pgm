"""treeSampler.py

2D square lattice Gaussian MRF using tree sampling.

"""
import numpy as np
from scipy import misc
from numpy.random import random_sample
import time

def main():
    # Number of particles - N, number of iterations of PGAS - R
    R = 1000
    
    # Initialize arrays/matrices
    nx = 10
    ny = 10
    fileName = './' + str(nx) + 'x' + str(ny) + 'TS.txt'
    
    # Load data
    Y = np.loadtxt('./inputData/' + str(nx) + 'x' + str(ny) + 'measurements.txt',delimiter=',')
    Y = Y.reshape( (nx, ny) )
    interactionPrec = np.loadtxt('./inputData/' + str(nx) + 'x' + str(ny) + 'intPrecision.txt',delimiter=',')
    observationPrec = np.loadtxt('./inputData/' + str(nx) + 'x' + str(ny) + 'obsPrecision.txt',delimiter=',')

    # MCMC initializations
    trajectoryMCMC = np.zeros( (nx,ny) )
    
    # Calculate indices, spiral shape
    if np.mod(ny, 2) == 0:
        nrPickInd = ny
        row = 0
        col = ny
        pickCol = 0
        pickRow = -1
    else:
        nrPickInd = nx
        row = -1
        col = 0
        pickCol = 1
        pickRow = 0
    iterations = 1
    index1 = np.zeros( (nx*ny), np.int )
    iterIndex = 0
    while nrPickInd > 0:
        if pickCol != 0:
            if pickCol == 1:
                for iPick in range(nrPickInd):
                    row = row + 1
                    tempFlatIndex = ravel_multi_index((row,col), (nx,ny))
                    index1[iterIndex] = tempFlatIndex
                    iterIndex = iterIndex + 1
                nrPickInd = ny - iterations
                pickCol = 0
                pickRow = 1
            else:
                for iPick in range(nrPickInd):
                    row = row - 1
                    tempFlatIndex = ravel_multi_index((row,col), (nx,ny))
                    index1[iterIndex] = tempFlatIndex
                    iterIndex = iterIndex + 1
                nrPickInd = ny - iterations
                pickCol = 0
                pickRow = -1
        else:
            if pickRow == 1:
                for iPick in range(nrPickInd):
                    col = col + 1
                    tempFlatIndex = ravel_multi_index((row,col), (nx,ny))
                    index1[iterIndex] = tempFlatIndex
                    iterIndex = iterIndex + 1
                nrPickInd = nx - iterations
                pickCol = -1
                pickRow = 0
            else:
                for iPick in range(nrPickInd):
                    col = col - 1
                    tempFlatIndex = ravel_multi_index((row,col), (nx,ny))
                    index1[iterIndex] = tempFlatIndex
                    iterIndex = iterIndex + 1
                nrPickInd = nx - iterations
                pickCol = 1
                pickRow = 0
        iterations = iterations + 1
    index1 = index1[0:iterIndex]
    # Index 2
    if np.mod(ny, 2) == 0:
        nrPickInd = nx-1
        row = nx
        col = ny-1
        pickCol = -1
        pickRow = 0
    else:
        nrPickInd = ny-1
        row = 0
        col = ny
        pickCol = 0
        pickRow = -1
    iterations = 2
    index2 = np.zeros( (nx*ny - iterIndex), np.int )
    iterIndex = 0
    while nrPickInd > 0:
        if pickCol != 0:
            if pickCol == 1:
                for iPick in range(nrPickInd):
                    row = row + 1
                    tempFlatIndex = ravel_multi_index((row,col), (nx,ny))
                    index2[iterIndex] = tempFlatIndex
                    iterIndex = iterIndex + 1
                nrPickInd = ny - iterations
                pickCol = 0
                pickRow = 1
            else:
                for iPick in range(nrPickInd):
                    row = row - 1
                    tempFlatIndex = ravel_multi_index((row,col), (nx,ny))
                    index2[iterIndex] = tempFlatIndex
                    iterIndex = iterIndex + 1
                nrPickInd = ny - iterations
                pickCol = 0
                pickRow = -1
        else:
            if pickRow == 1:
                for iPick in range(nrPickInd):
                    col = col + 1
                    tempFlatIndex = ravel_multi_index((row,col), (nx,ny))
                    index2[iterIndex] = tempFlatIndex
                    iterIndex = iterIndex + 1
                nrPickInd = nx - iterations
                pickCol = -1
                pickRow = 0
            else:
                for iPick in range(nrPickInd):
                    col = col - 1
                    tempFlatIndex = ravel_multi_index((row,col), (nx,ny))
                    index2[iterIndex] = tempFlatIndex
                    iterIndex = iterIndex + 1
                nrPickInd = nx - iterations
                pickCol = 1
                pickRow = 0
        iterations = iterations + 1
    #print index1
    #print index2
    #raw_input()
    # New file, print initial info, first line
    f = open(fileName, 'w')
    f.write('nx ny\n')
    f.write(str(nx) + ' ' + str(ny) + '\n')
    f.write('iterNr flat2Darray \n')
    f.write(str(0) + ' ')
    np.savetxt(f, trajectoryMCMC.reshape( (1, nx*ny) ))
    f.close()

    # ---------------------
    #      Tree sampling
    # ---------------------
    for iMCMC in np.arange(1,R):
        print iMCMC

        # --------------------
        #      FIRST CHAIN
        # --------------------
        
        # Initializations
        obsMean1 = np.zeros( len(index1) )
        obsCov1 = np.zeros( len(index1) )
        msgMean1 = np.zeros( len(index1)-1 )
        msgCov1 = np.zeros( len(index1)-1  )

        # Forward filtering
        for iFilt in range(len(index1)-1):
            # Node potential
            ix, iy = unravel_index(index1[iFilt], (nx,ny) )
            observationVar =  1 / observationPrec[index1[iFilt]]

            # Generate
            tempAvg = Y[ix,iy]
            tempVar = observationVar
            if ix > 0:
                if ravel_multi_index((ix-1,iy), (nx,ny)) in index2:
                    indN =  ravel_multi_index((ix-1, iy), (nx, ny))
                    interactionVar = 1 / interactionPrec[index1[iFilt],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix-1,iy]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if ix < nx-1:
                if ravel_multi_index((ix+1,iy), (nx,ny)) in index2:
                    indN =  ravel_multi_index((ix+1, iy), (nx, ny))
                    interactionVar = 1 / interactionPrec[index1[iFilt],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix+1,iy]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if iy > 0:
                if ravel_multi_index((ix,iy-1), (nx,ny)) in index2:
                    indN =  ravel_multi_index((ix, iy-1), (nx, ny))
                    interactionVar = 1 / interactionPrec[index1[iFilt],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy-1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if iy < ny-1:
                if ravel_multi_index((ix,iy+1), (nx,ny)) in index2:
                    indN =  ravel_multi_index((ix, iy+1), (nx, ny))
                    interactionVar = 1 / interactionPrec[index1[iFilt],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy+1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            obsMean1[iFilt] = tempAvg
            obsCov1[iFilt] = tempVar

            if iFilt > 0:
                tempAvg = (msgCov1[iFilt-1]*tempAvg + tempVar*msgMean1[iFilt-1]) / ( tempVar + msgCov1[iFilt-1] )
                tempVar = tempVar * msgCov1[iFilt-1] / (tempVar + msgCov1[iFilt-1])

            interactionVar = 1 / interactionPrec[index1[iFilt], index1[iFilt+1]]
            msgMean1[iFilt] = tempAvg
            msgCov1[iFilt] = tempVar + interactionVar

        # Node potential final node
        ix, iy = unravel_index(index1[-1], (nx,ny) )
        observationVar =  1 / observationPrec[index1[iFilt]]
        tempAvg = Y[ix,iy]
        tempVar = observationVar
        if ix > 0:
            if ravel_multi_index((ix-1,iy), (nx,ny)) in index2:
                indN =  ravel_multi_index((ix-1, iy), (nx, ny))
                interactionVar = 1 / interactionPrec[index1[iFilt],indN]
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix-1,iy]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
        if ix < nx-1:
            if ravel_multi_index((ix+1,iy), (nx,ny)) in index2:
                indN =  ravel_multi_index((ix+1, iy), (nx, ny))
                interactionVar = 1 / interactionPrec[index1[iFilt],indN]
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix+1,iy]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
        if iy > 0:
            if ravel_multi_index((ix,iy-1), (nx,ny)) in index2:
                indN =  ravel_multi_index((ix, iy-1), (nx, ny))
                interactionVar = 1 / interactionPrec[index1[iFilt],indN]
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy-1]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
        if iy < ny-1:
            if ravel_multi_index((ix,iy+1), (nx,ny)) in index2:
                indN =  ravel_multi_index((ix, iy+1), (nx, ny))
                interactionVar = 1 / interactionPrec[index1[iFilt],indN]
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy+1]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
        obsMean1[-1] = tempAvg
        obsCov1[-1] = tempVar

        # Backward sampling
        for iBack in range(len(index1))[::-1]:
            ix, iy = unravel_index(index1[iBack], (nx,ny) )
            tempAvg = obsMean1[iBack]
            tempVar = obsCov1[iBack]

            if iBack > 0:
                tempAvg = (msgCov1[iBack-1]*tempAvg + tempVar*msgMean1[iBack-1]) / ( tempVar + msgCov1[iBack-1] )
                tempVar = tempVar * msgCov1[iBack-1] / (tempVar + msgCov1[iBack-1])
            if iBack < len(index1)-1:
                interactionVar = 1 / interactionPrec[index1[iBack],index1[iBack+1]]
                ixN, iyN = unravel_index(index1[iBack+1], (nx,ny) )
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ixN,iyN]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            trajectoryMCMC[ix,iy] = tempAvg + np.sqrt(tempVar) * np.random.randn()
        
        # --------------------
        #      SECOND CHAIN
        # --------------------
        
        # Initializations
        obsMean2 = np.zeros( len(index2) )
        obsCov2 = np.zeros( len(index2) )
        msgMean2 = np.zeros( len(index2)-1 )
        msgCov2 = np.zeros( len(index2)-1  )

        # Forward filtering
        for iFilt in range(len(index2)-1):
            # Node potential
            ix, iy = unravel_index(index2[iFilt], (nx,ny) )
            observationVar =  1 / observationPrec[index2[iFilt]]

            # Generate
            tempAvg = Y[ix,iy]
            tempVar = observationVar
            if ix > 0:
                if ravel_multi_index((ix-1,iy), (nx,ny)) in index1:
                    indN =  ravel_multi_index((ix-1, iy), (nx, ny))
                    interactionVar = 1 / interactionPrec[index2[iFilt],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix-1,iy]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if ix < nx-1:
                if ravel_multi_index((ix+1,iy), (nx,ny)) in index1:
                    indN =  ravel_multi_index((ix+1, iy), (nx, ny))
                    interactionVar = 1 / interactionPrec[index2[iFilt],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix+1,iy]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if iy > 0:
                if ravel_multi_index((ix,iy-1), (nx,ny)) in index1:
                    indN =  ravel_multi_index((ix, iy-1), (nx, ny))
                    interactionVar = 1 / interactionPrec[index2[iFilt],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy-1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if iy < ny-1:
                if ravel_multi_index((ix,iy+1), (nx,ny)) in index1:
                    indN =  ravel_multi_index((ix, iy+1), (nx, ny))
                    interactionVar = 1 / interactionPrec[index2[iFilt],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy+1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            obsMean2[iFilt] = tempAvg
            obsCov2[iFilt] = tempVar

            if iFilt > 0:
                tempAvg = (msgCov2[iFilt-1]*tempAvg + tempVar*msgMean2[iFilt-1]) / ( tempVar + msgCov2[iFilt-1] )
                tempVar = tempVar * msgCov2[iFilt-1] / (tempVar + msgCov2[iFilt-1])

            interactionVar = 1 / interactionPrec[index2[iFilt], index2[iFilt+1]]
            msgMean2[iFilt] = tempAvg
            msgCov2[iFilt] = tempVar + interactionVar

        # Node potential final node
        ix, iy = unravel_index(index2[-1], (nx,ny) )
        observationVar =  1 / observationPrec[index2[iFilt]]
        tempAvg = Y[ix,iy]
        tempVar = observationVar
        if ix > 0:
            if ravel_multi_index((ix-1,iy), (nx,ny)) in index1:
                indN =  ravel_multi_index((ix-1, iy), (nx, ny))
                interactionVar = 1 / interactionPrec[index2[iFilt],indN]
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix-1,iy]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
        if ix < nx-1:
            if ravel_multi_index((ix+1,iy), (nx,ny)) in index1:
                indN =  ravel_multi_index((ix+1, iy), (nx, ny))
                interactionVar = 1 / interactionPrec[index2[iFilt],indN]
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix+1,iy]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
        if iy > 0:
            if ravel_multi_index((ix,iy-1), (nx,ny)) in index1:
                indN =  ravel_multi_index((ix, iy-1), (nx, ny))
                interactionVar = 1 / interactionPrec[index2[iFilt],indN]
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy-1]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
        if iy < ny-1:
            if ravel_multi_index((ix,iy+1), (nx,ny)) in index1:
                indN =  ravel_multi_index((ix, iy+1), (nx, ny))
                interactionVar = 1 / interactionPrec[index2[iFilt],indN]
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy+1]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
        obsMean2[-1] = tempAvg
        obsCov2[-1] = tempVar

        # Backward sampling
        for iBack in range(len(index2))[::-1]:
            ix, iy = unravel_index(index2[iBack], (nx,ny) )
            tempAvg = obsMean2[iBack]
            tempVar = obsCov2[iBack]

            if iBack > 0:
                tempAvg = (msgCov2[iBack-1]*tempAvg + tempVar*msgMean2[iBack-1]) / ( tempVar + msgCov2[iBack-1] )
                tempVar = tempVar * msgCov2[iBack-1] / (tempVar + msgCov2[iBack-1])
            if iBack < len(index2)-1:
                interactionVar = 1 / interactionPrec[index2[iBack],index2[iBack+1]]
                ixN, iyN = unravel_index(index2[iBack+1], (nx,ny) )
                tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ixN,iyN]) / ( tempVar + interactionVar )
                tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            trajectoryMCMC[ix,iy] = tempAvg + np.sqrt(tempVar) * np.random.randn()

	        

        # Save current iteration
        f = open(fileName, 'a')
        f.write(str(iMCMC) + ' ')
        np.savetxt(f, trajectoryMCMC.reshape( (1, nx*ny) ))
        f.close()

    

def discreteSampling(weights, domain, nrSamples):
    bins = np.cumsum(weights)
    return domain[np.digitize(random_sample(nrSamples), bins)]

def ravel_multi_index(coord, shape):
    return coord[0] * shape[1] + coord[1]

def unravel_index(coord, shape):
    iy = np.remainder(coord, shape[1])
    ix = (coord - iy) / shape[1]
    return ix, iy
    
if __name__ == "__main__":
    main()
