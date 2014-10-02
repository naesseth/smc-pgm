"""pgas.py

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
    N = 15
    R = 1000
    # Initialize arrays/matrices
    nx = 10
    ny = 10
    fileName = './' + str(nx) + 'x' + str(ny) + 'PGASN' + str(N) + '.txt'
    index = np.zeros( (nx*ny), np.int )

    # Load data
    Y = np.loadtxt('./inputData/' + str(nx) + 'x' + str(ny) + 'measurements.txt',delimiter=',')
    Y = Y.reshape( (nx, ny) )
    interactionPrec = np.loadtxt('./inputData/' + str(nx) + 'x' + str(ny) + 'intPrecision.txt',delimiter=',')
    observationPrec = np.loadtxt('./inputData/' + str(nx) + 'x' + str(ny) + 'obsPrecision.txt',delimiter=',')
    
    # MCMC initializations, :,0 - mean, :,1 - variance
    trajectoryMCMC = np.zeros( len(index) )
    
    # Choose sequential order (spiral, snake, ...)
    order = 'snake'
    # Calculate indices, snake shape (right left right ...)
    if order == 'snake':
        for iIter in range(nx):
            if np.mod(iIter, 2) == 0:
                index[ny*iIter:ny*(iIter+1)] = np.arange(ny*iIter,ny*(iIter+1))
            else:
                index[ny*iIter:ny*(iIter+1)] = np.arange(ny*iIter,ny*(iIter+1))[::-1]
    
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
        # ---------------
        #      CSMC
        # ---------------
        
        # CSMC initializations
        tempWeights = np.zeros( N )
        ancestors = np.zeros( (N, len(index)), np.int )
        trajectorySMC = np.zeros( (N, len(index)) )
        
        #CSMC first iteration
        tempAvg = Y[0,0]
        observationVar =  1 / observationPrec[0]
        tempVar = observationVar
        currentEstimate = tempAvg + np.sqrt(tempVar) * np.random.randn(N-1)
        trajectorySMC[:N-1, 0] = currentEstimate
        # Condition on last MCMC trajectory
        trajectorySMC[-1,0] = trajectoryMCMC[0]
       
        # CSMC main loop
        for iSMC in np.arange(1,len(index)):
            # Define index sets used to speed up execution
            p0 = np.maximum(iSMC-2*ny,0)
            c0 = np.minimum(iSMC+2*ny,nx*ny-1)
            indexParticle = index[p0:iSMC]
            indexCondition = index[iSMC:c0]
            indexSortedParticle = np.argsort(indexParticle)
            indexSortedCondition = np.argsort(indexCondition)

            #startSMC = time.time()
            
            # Node potential
            ix, iy = unravel_index(index[iSMC], (nx,ny) )
            observationVar =  1 / observationPrec[index[iSMC]]
            tempAvg = Y[ix,iy]
            tempVar = observationVar
            tempParticleAvg = np.zeros( N )
            tempParticleVar = np.zeros( N )
            tempWeights = np.ones( N )
            # Draw samples from kernel M(a,x)
            #startW = time.time()
            for iParticle in range(N):
                tempParticleAvg[iParticle] = tempAvg
                tempParticleVar[iParticle] = tempVar
                if ix > 0:
                    indPrev = ravel_multi_index((ix-1,iy), (nx,ny))
                    if  indPrev in index[:iSMC]:
                        parentInd = np.searchsorted(indexParticle[indexSortedParticle], indPrev)
                        parentInd = indexSortedParticle[parentInd]+p0
			interactionVar = 1 / interactionPrec[index[iSMC],index[parentInd]]
                        tempWeights[iParticle] = tempWeights[iParticle] * np.exp(-0.5 * (tempParticleAvg[iParticle] - trajectorySMC[iParticle,parentInd])**2 / ( tempParticleVar[iParticle] + interactionVar ) )
                        tempParticleAvg[iParticle] = (interactionVar*tempParticleAvg[iParticle] + tempParticleVar[iParticle]*trajectorySMC[iParticle,parentInd]) / ( tempParticleVar[iParticle] + interactionVar )
                        tempParticleVar[iParticle] = tempParticleVar[iParticle] * interactionVar / (interactionVar + tempParticleVar[iParticle])
                        
                if iy > 0:
                    indPrev = ravel_multi_index((ix,iy-1), (nx,ny))
                    if indPrev in index[:iSMC]:
                        parentInd = np.searchsorted(indexParticle[indexSortedParticle], indPrev)
                        parentInd = indexSortedParticle[parentInd]+p0
                        interactionVar = 1 / interactionPrec[index[iSMC],index[parentInd]]
                        tempWeights[iParticle] = tempWeights[iParticle] * np.exp(-0.5 * (tempParticleAvg[iParticle] - trajectorySMC[iParticle,parentInd])**2 / ( tempParticleVar[iParticle] + interactionVar ) )
                        tempParticleAvg[iParticle] = (interactionVar*tempParticleAvg[iParticle] + tempParticleVar[iParticle]*trajectorySMC[iParticle,parentInd]) / ( tempParticleVar[iParticle] + interactionVar )
                        tempParticleVar[iParticle] = tempParticleVar[iParticle] * interactionVar / (interactionVar + tempParticleVar[iParticle])
                if iy < ny-1:
                    indPrev = ravel_multi_index((ix,iy+1), (nx,ny))
                    if indPrev in index[:iSMC]:
                        parentInd = np.searchsorted(indexParticle[indexSortedParticle], indPrev)
                        parentInd = indexSortedParticle[parentInd]+p0
                        interactionVar = 1 / interactionPrec[index[iSMC],index[parentInd]]
                        tempWeights[iParticle] = tempWeights[iParticle] * np.exp(-0.5 * (tempParticleAvg[iParticle] - trajectorySMC[iParticle,parentInd])**2 / ( tempParticleVar[iParticle] + interactionVar ) )
                        tempParticleAvg[iParticle] = (interactionVar*tempParticleAvg[iParticle] + tempParticleVar[iParticle]*trajectorySMC[iParticle,parentInd]) / ( tempParticleVar[iParticle] + interactionVar )
                        tempParticleVar[iParticle] = tempParticleVar[iParticle] * interactionVar / (interactionVar + tempParticleVar[iParticle])
                tempWeights[iParticle] = tempWeights[iParticle] * np.sqrt(2 * np.pi * tempParticleVar[iParticle])

            ancestors[:N-1,iSMC] = discreteSampling( tempWeights / np.sum( tempWeights ), np.arange(N), N-1)
            for iParticle in range(N-1):
                currentEstimate = tempParticleAvg[ancestors[iParticle,iSMC]] + np.sqrt(tempParticleVar[ancestors[iParticle,iSMC]]) * np.random.randn()
                trajectorySMC[iParticle, iSMC] = currentEstimate
            
            # Conditioning
            trajectorySMC[-1,iSMC] =  trajectoryMCMC[iSMC]
            # Ancestor sampling
            tempDistAS = np.ones( N )
            interactionVar = 1 / interactionPrec[index[iSMC-1],index[iSMC]]
            tempDistAS = tempDistAS * np.exp( (- 1 / (2*interactionVar) ) * (trajectorySMC[:,iSMC-1] - trajectoryMCMC[iSMC])**2 )
            for iPrev in range(ny):
                if ix > 0:
                    indPrev1 = ravel_multi_index((ix-1,iPrev), (nx,ny))
                    indPrev2 = ravel_multi_index((ix,iPrev), (nx,ny))
                    if indPrev2 in index[iSMC:] and indPrev1 in index[:iSMC] and (indPrev2 != index[iSMC] or indPrev1 != index[iSMC-1]):
                        particleInd = np.searchsorted(indexParticle[indexSortedParticle], indPrev1)
                        particleInd = indexSortedParticle[particleInd]+p0
                        conditionInd = np.searchsorted(indexCondition[indexSortedCondition], indPrev2)+iSMC
                        interactionVar = 1 / interactionPrec[index[particleInd],index[conditionInd]]
                        tempDistAS = tempDistAS * np.exp( (- 1 / (2*interactionVar) ) * (trajectorySMC[:,particleInd] - trajectoryMCMC[conditionInd])**2 )
                if ix < nx-1:
                    indPrev1 = ravel_multi_index((ix,iPrev), (nx,ny))
                    indPrev2 = ravel_multi_index((ix+1,iPrev), (nx,ny))
                    if indPrev2 in index[iSMC:] and indPrev1 in index[:iSMC] and (indPrev2 != index[iSMC] or indPrev1 != index[iSMC-1]):
                        particleInd = np.searchsorted(indexParticle[indexSortedParticle], indPrev1)
                        particleInd = indexSortedParticle[particleInd]+p0
                        conditionInd = np.searchsorted(indexCondition[indexSortedCondition], indPrev2)+iSMC
                        interactionVar = 1 / interactionPrec[index[particleInd],index[conditionInd]]
                        tempDistAS = tempDistAS * np.exp( (- 1 / (2*interactionVar) ) * (trajectorySMC[:,particleInd] - trajectoryMCMC[conditionInd])**2 )
            tempDistAS = tempDistAS / np.sum( tempDistAS )
            ancestors[-1,iSMC] = discreteSampling( tempDistAS, np.arange(N), 1)
            # Set x = {x1:t-1,xt}, calc. weights
            trajectorySMC[:,:iSMC] = trajectorySMC[ancestors[:,iSMC].reshape( N ).astype(int),:iSMC]

        # ---------------
        #    MCMC step
        # ---------------      
        indMCMC = discreteSampling(np.ones( N ) / N, np.arange(N), 1)
        trajectoryMCMC = trajectorySMC[indMCMC[0],:]
        # Save current iteration
        f = open(fileName, 'a')
        f.write(str(iMCMC) + ' ')
        np.savetxt(f, trajectoryMCMC[index].reshape( (1, nx*ny) ))
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
