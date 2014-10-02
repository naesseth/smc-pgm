"""pgasPartialBlocking.py

2D square lattice Gaussian MRF using Particle Gibbs with Ancestor sampling.

"""
import numpy as np
from scipy import misc
from numpy.random import random_sample
import time

def main():
    # Size of problem
    M = 10
    
    # Number of particles - N, number of iterations of PGAS - R
    N = 20
    R = 1000
    
    # Initialize arrays/matrices
    nx = 10
    ny = 10
    fileName = './' + str(nx) + 'x' + str(ny) + 'PGASpbN' + str(N) + '.txt'
    
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

    # ---------------
    #      PMCMC
    # ---------------
    for iMCMC in np.arange(1,R):
        print iMCMC

        # --------------------
        #      FIRST CHAIN
        # --------------------
        
        # ---------------
        #      CSMC
        # ---------------
        
        # CSMC initializations
        tempWeights = np.zeros( N )
        ancestors1 = np.zeros( (N, len(index1)), np.int )
        trajectorySMC1 = np.zeros( (N, len(index1)) )
       
        # CSMC main loop
        for iSMC in np.arange(0,len(index1)):
            # Node potential
            ix, iy = unravel_index(index1[iSMC], (nx,ny) )
            observationVar =  1 / observationPrec[index1[iSMC]]
            tempAvg = Y[ix,iy]
            tempVar = observationVar
            if ix > 0:
                if ravel_multi_index((ix-1,iy), (nx,ny)) in index2:
                    indN =  ravel_multi_index((ix-1, iy), (nx, ny))
                    interactionVar = 1 / interactionPrec[index1[iSMC],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix-1,iy]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if ix < nx-1:
                if ravel_multi_index((ix+1,iy), (nx,ny)) in index2:
                    indN =  ravel_multi_index((ix+1, iy), (nx, ny))
                    interactionVar = 1 / interactionPrec[index1[iSMC],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix+1,iy]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if iy > 0:
                if ravel_multi_index((ix,iy-1), (nx,ny)) in index2:
                    indN =  ravel_multi_index((ix, iy-1), (nx, ny))
                    interactionVar = 1 / interactionPrec[index1[iSMC],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy-1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if iy < ny-1:
                if ravel_multi_index((ix,iy+1), (nx,ny)) in index2:
                    indN =  ravel_multi_index((ix, iy+1), (nx, ny))
                    interactionVar = 1 / interactionPrec[index1[iSMC],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy+1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
                    
            tempParticleAvg = np.zeros( N )
            tempParticleVar = np.zeros( N )
            tempWeights = np.ones( N )
            # Draw samples from kernel M(a,x)
            for iParticle in range(N):
                tempParticleAvg[iParticle] = tempAvg
                tempParticleVar[iParticle] = tempVar
                if iSMC > 0:
                    parentInd = iSMC-1
                    interactionVar = 1 / interactionPrec[index1[iSMC],index1[parentInd]]
                    tempWeights[iParticle] = tempWeights[iParticle] * np.exp(-0.5 * (tempParticleAvg[iParticle] - trajectorySMC1[iParticle,parentInd])**2 / ( tempParticleVar[iParticle] + interactionVar ) )
                    tempParticleAvg[iParticle] = (interactionVar*tempParticleAvg[iParticle] + tempParticleVar[iParticle]*trajectorySMC1[iParticle,parentInd]) / ( tempParticleVar[iParticle] + interactionVar )
                    tempParticleVar[iParticle] = tempParticleVar[iParticle] * interactionVar / (interactionVar + tempParticleVar[iParticle])
                tempWeights[iParticle] = tempWeights[iParticle] * np.sqrt(2 * np.pi * tempParticleVar[iParticle])
            if iSMC > 0:
                ancestors1[:N-1,iSMC] = discreteSampling( tempWeights / np.sum( tempWeights ), np.arange(N), N-1)
                for iParticle in range(N-1):
                    currentEstimate = tempParticleAvg[ancestors1[iParticle,iSMC]] + np.sqrt(tempParticleVar[ancestors1[iParticle,iSMC]]) * np.random.randn()
                    trajectorySMC1[iParticle, iSMC] = currentEstimate
                # Conditioning
                trajectorySMC1[-1,iSMC] =  trajectoryMCMC[ix,iy]
                # Ancestor sampling
                tempDistAS = np.ones( N )
                interactionVar = 1 / interactionPrec[index1[iSMC-1],index1[iSMC]]
                tempDistAS = tempDistAS * np.exp( (- 1 / (2*interactionVar) ) * (trajectorySMC1[:,iSMC-1] - trajectoryMCMC[ix,iy])**2 )
                tempDistAS = tempDistAS / np.sum( tempDistAS )
                ancestors1[-1,iSMC] = discreteSampling( tempDistAS, np.arange(N), 1)
                trajectorySMC1[:,:iSMC] = trajectorySMC1[ancestors1[:,iSMC].reshape( N ).astype(int),:iSMC]

            else:
                for iParticle in range(N-1):
                    currentEstimate = tempParticleAvg[iParticle] + np.sqrt(tempParticleVar[iParticle]) * np.random.randn()
                    trajectorySMC1[iParticle, iSMC] = currentEstimate
                trajectorySMC1[-1,iSMC] =  trajectoryMCMC[ix,iy]


        # ---------------
        #    MCMC step 1
        # ---------------      
        indMCMC = discreteSampling(np.ones( N ) / N, np.arange(N), 1)
        trajectoryMCMC.flat[index1] = trajectorySMC1[indMCMC[0],:]
	
        
        # --------------------
        #      SECOND CHAIN
        # --------------------
        
        # ---------------
        #      CSMC
        # ---------------
        
        # CSMC initializations
        tempWeights = np.zeros( N )
        ancestors2 = np.zeros( (N, len(index2)), np.int )
        trajectorySMC2 = np.zeros( (N, len(index2)) )
       
        # CSMC main loop
        for iSMC in np.arange(0,len(index2)):
            # Node potential
            ix, iy = unravel_index(index2[iSMC], (nx,ny) )
            observationVar =  1 / observationPrec[index2[iSMC]]
            tempAvg = Y[ix,iy]
            tempVar = observationVar
            if ix > 0:
                if ravel_multi_index((ix-1,iy), (nx,ny)) in index1:
                    indN =  ravel_multi_index((ix-1, iy), (nx, ny))
                    interactionVar = 1 / interactionPrec[index2[iSMC],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix-1,iy]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if ix < nx-1:
                if ravel_multi_index((ix+1,iy), (nx,ny)) in index1:
                    indN =  ravel_multi_index((ix+1, iy), (nx, ny))
                    interactionVar = 1 / interactionPrec[index2[iSMC],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix+1,iy]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if iy > 0:
                if ravel_multi_index((ix,iy-1), (nx,ny)) in index1:
                    indN =  ravel_multi_index((ix, iy-1), (nx, ny))
                    interactionVar = 1 / interactionPrec[index2[iSMC],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy-1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            if iy < ny-1:
                if ravel_multi_index((ix,iy+1), (nx,ny)) in index1:
                    indN =  ravel_multi_index((ix, iy+1), (nx, ny))
                    interactionVar = 1 / interactionPrec[index2[iSMC],indN]
                    tempAvg = (interactionVar*tempAvg + tempVar*trajectoryMCMC[ix,iy+1]) / ( tempVar + interactionVar )
                    tempVar = tempVar * interactionVar / (interactionVar + tempVar)
            tempParticleAvg = np.zeros( N )
            tempParticleVar = np.zeros( N )
            tempWeights = np.ones( N )
            # Draw samples from kernel M(a,x)
            for iParticle in range(N):
                tempParticleAvg[iParticle] = tempAvg
                tempParticleVar[iParticle] = tempVar
                if iSMC > 0:
                    parentInd = iSMC-1
                    interactionVar = 1 / interactionPrec[index2[iSMC],index2[parentInd]]
                    tempWeights[iParticle] = tempWeights[iParticle] * np.exp(-0.5 * (tempParticleAvg[iParticle] - trajectorySMC2[iParticle,parentInd])**2 / ( tempParticleVar[iParticle] + interactionVar ) )
                    tempParticleAvg[iParticle] = (interactionVar*tempParticleAvg[iParticle] + tempParticleVar[iParticle]*trajectorySMC2[iParticle,parentInd]) / ( tempParticleVar[iParticle] + interactionVar )
                    tempParticleVar[iParticle] = tempParticleVar[iParticle] * interactionVar / (interactionVar + tempParticleVar[iParticle])
                tempWeights[iParticle] = tempWeights[iParticle] * np.sqrt(2 * np.pi * tempParticleVar[iParticle])

            if iSMC > 0:
                ancestors2[:N-1,iSMC] = discreteSampling( tempWeights / np.sum( tempWeights ), np.arange(N), N-1)
                for iParticle in range(N-1):
                    currentEstimate = tempParticleAvg[ancestors2[iParticle,iSMC]] + np.sqrt(tempParticleVar[ancestors2[iParticle,iSMC]]) * np.random.randn()
                    trajectorySMC2[iParticle, iSMC] = currentEstimate
            
                # Conditioning
                trajectorySMC2[-1,iSMC] =  trajectoryMCMC[ix,iy]
                # Ancestor sampling
           
                tempDistAS = np.ones( N )
                interactionVar = 1 / interactionPrec[index2[iSMC-1],index2[
iSMC]]
                tempDistAS = tempDistAS * np.exp( (- 1 / (2*interactionVar) ) * (trajectorySMC2[:,iSMC-1] - trajectoryMCMC[ix,iy])**2 )
                tempDistAS = tempDistAS / np.sum( tempDistAS )
                ancestors2[-1,iSMC] = discreteSampling( tempDistAS, np.arange(N), 1)
                trajectorySMC2[:,:iSMC] = trajectorySMC2[ancestors2[:,iSMC].reshape( N ).astype(int),:iSMC]
            else:
                for iParticle in range(N-1):
                    currentEstimate = tempParticleAvg[iParticle] + np.sqrt(tempParticleVar[iParticle]) * np.random.randn()
                    trajectorySMC2[iParticle, iSMC] = currentEstimate
                trajectorySMC2[-1,iSMC] =  trajectoryMCMC[ix,iy]

        # ---------------
        #    MCMC step 2
        # ---------------      
        indMCMC = discreteSampling(np.ones( N ) / N, np.arange(N), 1)
        trajectoryMCMC.flat[index2] = trajectorySMC2[indMCMC[0],:]
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
