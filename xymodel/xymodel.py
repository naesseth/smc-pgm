"""

Module for partition function estimation of square lattice XY model
with free boundary conditions using Monte Carlo methods. 
Focus is Sequential Monte Carlo (SMC) methods.

Current supported methods:
Sequential Monte Carlo methods for graphical models
-- Naesseth et. al. 2014
Annealed Importance Sampling
-- Neal, 2001
Importance Sampling

"""

import numpy as np
import helpfunctions as hlp

class classicalXY:
    """ 
    Square lattice classical XY model (free) base class defined by
    
    .. math::
        p( \mathbf{x} ) \propto \exp -\beta H(\mathbf{x}),
        
    where the Hamiltonian is given by
    
    .. math::
        H(\mathbf{x}) = - J \sum_{(i,j) \in \mathcal{E}} \cos (x_i - x_j)
    
    Parameters
    ----------
    J : float
        The interaction parameters J of H(x).
    sz : int
        The size of the square lattice model.
    """
    def __init__(self, sz, J):
        assert type(sz) is int, "sz must be int"
         
        self.J = J
        self.sz = sz
    
    def smc(self, order, N, NT=1.1, resamp='mult', verbose=False):
        """
        SMC algorithm to estimate (log)Z of the classical XY model with 
        free boundary conditions.
        
        Parameters
        ----------
        order : 1-D array_like
            The order in which to add the random variables x, flat index.
        N : int
            The number of particles used to estimate Z.
        NT : int
            Threshold for ESS-based resampling (0,1] or 1.1. Resample if ESS < NT*N (NT=1.1, resample every time)
        resamp : string
            Type of resampling scheme {mult, res, strat, sys}.
        verbose : bool
            Output changed to logZk, xMean, ESS.
        
        Output
        ------
        logZ : float
            Estimate of (log)Z in double precision.
        """
        # Init variables
        nx = self.sz
        ny = self.sz
        logZ = np.zeros( nx*ny )
        indSorted = order.argsort()
        orderSorted = order[indSorted]
        # SMC specific
        trajectory = np.zeros( (N, len(order)) )
        ancestors = np.zeros( N, np.int )
        nu = np.zeros( N )
        tempNu = np.zeros( N )
        ess = np.zeros( len(order)-1 )
        iter = 0

        # -------
        #   SMC
        # -------
        # First iteration
        ix, iy = hlp.unravel_index( order[0], (nx,ny) )
        tempMean = 0.
        tempDispersion = 0.
        trajectory[:,0] = np.random.vonmises(tempMean, tempDispersion, N)
        # Log-trick update of adjustment multipliers and logZ
        tempDispersion = np.zeros(N)

        for iSMC in range( 1, len(order) ):
            # Resampling with log-trick update
            nu += np.log(2 * np.pi * np.i0(tempDispersion))
            nuMax = np.max(nu)
            tempNu = np.exp( nu - nuMax )
            c = np.sum(tempNu)
            tempNu /= c
            ess[iSMC-1] = 1 / (np.sum(tempNu**2))

            if ess[iSMC-1] < NT*float(N):
                nu = np.exp( nu - nuMax )
                if iter > 0:
                    logZ[iter] = logZ[iter-1] + nuMax + np.log( np.sum(nu) ) - np.log(N)
                else:
                    logZ[iter] = nuMax + np.log( np.sum(nu) ) - np.log(N)
                c = np.sum(nu)
                nu /= c
                ancestors = hlp.resampling( nu, scheme=resamp )
                nu = np.zeros( N )
                trajectory[:,:iSMC] = trajectory[ancestors, :iSMC]
                iter += 1

            # Calculate optimal proposal and adjustment multipliers
            ix, iy = hlp.unravel_index( order[iSMC], (nx,ny) )
            tempMean = np.zeros( N )
            tempDispersion = np.zeros( N )
            if ix > 0:
                tempInd = hlp.ravel_multi_index( (ix-1,iy), (nx,ny) )
                if tempInd in order[:iSMC]:
                    kappa = self.J
                    tempIndSMC = indSorted[orderSorted.searchsorted(tempInd)]
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -trajectory[:,tempIndSMC] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -trajectory[:,tempIndSMC] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+trajectory[:,tempIndSMC] ) )
                    tempMean = -np.arctan2(Y ,X)
            else:
                tempInd = hlp.ravel_multi_index( (nx-1,iy), (nx,ny) )
                if tempInd in order[:iSMC]:
                    kappa = self.J
                    tempIndSMC = indSorted[orderSorted.searchsorted(tempInd)]
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -trajectory[:,tempIndSMC] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -trajectory[:,tempIndSMC] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+trajectory[:,tempIndSMC] ) )
                    tempMean = -np.arctan2(Y ,X)
            if ix < nx-1:
                tempInd = hlp.ravel_multi_index( (ix+1,iy), (nx,ny) )
                if tempInd in order[:iSMC]:
                    kappa = self.J
                    tempIndSMC = indSorted[orderSorted.searchsorted(tempInd)]
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -trajectory[:,tempIndSMC] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -trajectory[:,tempIndSMC] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+trajectory[:,tempIndSMC] ) )
                    tempMean = -np.arctan2(Y ,X)
            else:
                tempInd = hlp.ravel_multi_index( (0,iy), (nx,ny) )
                if tempInd in order[:iSMC]:
                    kappa = self.J
                    tempIndSMC = indSorted[orderSorted.searchsorted(tempInd)]
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -trajectory[:,tempIndSMC] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -trajectory[:,tempIndSMC] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+trajectory[:,tempIndSMC] ) )
                    tempMean = -np.arctan2(Y ,X)
            if iy > 0:
                tempInd = hlp.ravel_multi_index( (ix,iy-1), (nx,ny) )
                if tempInd in order[:iSMC]:
                    kappa = self.J
                    tempIndSMC = indSorted[orderSorted.searchsorted(tempInd)]
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -trajectory[:,tempIndSMC] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -trajectory[:,tempIndSMC] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+trajectory[:,tempIndSMC] ) )
                    tempMean = -np.arctan2(Y ,X)
            else:
                tempInd = hlp.ravel_multi_index( (ix,ny-1), (nx,ny) )
                if tempInd in order[:iSMC]:
                    kappa = self.J
                    tempIndSMC = indSorted[orderSorted.searchsorted(tempInd)]
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -trajectory[:,tempIndSMC] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -trajectory[:,tempIndSMC] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+trajectory[:,tempIndSMC] ) )
                    tempMean = -np.arctan2(Y ,X)
            if iy < ny-1:
                tempInd = hlp.ravel_multi_index( (ix,iy+1), (nx,ny) )
                if tempInd in order[:iSMC]:
                    kappa = self.J
                    tempIndSMC = indSorted[orderSorted.searchsorted(tempInd)]
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -trajectory[:,tempIndSMC] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -trajectory[:,tempIndSMC] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+trajectory[:,tempIndSMC] ) )
                    tempMean = -np.arctan2(Y ,X)
            else:
                tempInd = hlp.ravel_multi_index( (ix,0), (nx,ny) )
                if tempInd in order[:iSMC]:
                    kappa = self.J
                    tempIndSMC = indSorted[orderSorted.searchsorted(tempInd)]
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -trajectory[:,tempIndSMC] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -trajectory[:,tempIndSMC] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+trajectory[:,tempIndSMC] ) )
                    tempMean = -np.arctan2(Y ,X)
            for iParticle in range(N):
                trajectory[iParticle, iSMC] = hlp.vonmises(tempMean[iParticle], tempDispersion[iParticle])

        nu += np.log(2 * np.pi * np.i0(tempDispersion))
        nuMax = np.max(nu)
        nu = np.exp( nu - nuMax )
        logZ[iter] = logZ[iter-1] + nuMax + np.log( np.sum(nu) ) - np.log(N)
        
        if verbose:
            c = np.sum(nu)
            nu /= c
            trajMean = np.mean( (np.tile(nu, (len(order),1))).T*trajectory, axis=0 )
            return logZ, trajMean[order].reshape( (nx,ny) ), ess
        else:
            return logZ[iter]
        
    def importanceSampling(self, K):
        # Init variables
        nx = self.sz
        ny = self.sz
        logZ = 0.
        samples = np.zeros( (K, nx, ny) )
        evalPQ = np.zeros( K )
        
        for iX in range(nx):
            for iY in range(ny):
                kappa = 0.
                logZ += np.log( 2 * np.pi * np.i0(kappa) )
                samples[:, iX, iY] = np.random.vonmises(0, kappa, K)
        #print logZ

        for iSample in range(K):
            for iX in range(nx):
                for iY in range(ny):
                    curInd = hlp.ravel_multi_index( (iX, iY), (nx,ny) )
                    if iX > 0:
                        neiInd = hlp.ravel_multi_index( (iX-1, iY), (nx,ny) )
                        evalPQ[iSample] += self.J * np.cos( samples[iSample, iX, iY] - samples[iSample, iX-1, iY] ) 
                    else:
                        neiInd = hlp.ravel_multi_index( (nx-1, iY), (nx,ny) )
                        evalPQ[iSample] += self.J * np.cos( samples[iSample, iX, iY] - samples[iSample, nx-1, iY] ) 
                    if iY > 0:
                        neiInd = hlp.ravel_multi_index( (iX, iY-1), (nx,ny) )
                        evalPQ[iSample] += self.J * np.cos( samples[iSample, iX, iY] - samples[iSample, iX, iY-1] )
                    else:
                        neiInd = hlp.ravel_multi_index( (iX, ny-1), (nx,ny) )
                        evalPQ[iSample] += self.J * np.cos( samples[iSample, iX, iY] - samples[iSample, iX, ny-1] )                        
        
        evalMax = np.max( evalPQ )
        evalPQ = np.exp( evalPQ - evalMax )
        logZ += evalMax - np.log(K) + np.log( np.sum( evalPQ ) )
        return logZ
        
    def annealedImportanceSampling(self, path, N):
        """
        AIS algorithm to estimate (log)Z of the classical XY model with 
        free boundary conditions. fn is bias (external field) terms and
        f0 is the full unnormalized PDF.
        
        Parameters
        ----------
        path : 1-D array_like
            The path to anneal between fn and f0.
        N : int
            The number of annealed importance samples.
        
        Output
        ------
        logZ : float
            Estimate of (log)Z in double precision.
        """
        logZ = 0.
        logWeights = np.zeros( N )
        nx = self.sz
        ny = self.sz

        XY = np.zeros( (N, nx, ny) )
        for iPath in np.arange(len(path)-1):
            #print 'Path: ',iPath
            index = np.arange( nx*ny )
            #np.random.shuffle(index)
            for ind in index:
                iX, iY = hlp.unravel_index(ind, (nx,ny))
                # Sample from Markov kernel (Gibbs)
                tempMean = np.zeros( N )
                tempDispersion = np.zeros( N )
                curInd = ind
                if iX > 0:
                    neiInd = hlp.ravel_multi_index( (iX-1, iY), (nx,ny) )
                    kappa = path[iPath]*self.J
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -XY[:,iX-1, iY] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -XY[:,iX-1, iY] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+XY[:,iX-1, iY] ) )
                    tempMean = -np.arctan2(Y ,X)
                else:
                    neiInd = hlp.ravel_multi_index( (nx-1, iY), (nx,ny) )
                    kappa = path[iPath]*self.J
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -XY[:,nx-1, iY] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -XY[:,nx-1, iY] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+XY[:,nx-1, iY] ) )
                    tempMean = -np.arctan2(Y ,X)                            
                if iX < nx-1:
                    neiInd = hlp.ravel_multi_index( (iX+1, iY), (nx,ny) )
                    kappa = path[iPath]*self.J
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -XY[:,iX+1, iY] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -XY[:,iX+1, iY] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+XY[:,iX+1, iY] ) )
                    tempMean = -np.arctan2(Y ,X)
                else:
                    neiInd = hlp.ravel_multi_index( (0, iY), (nx,ny) )
                    kappa = path[iPath]*self.J
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -XY[:,0, iY] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -XY[:,0, iY] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+XY[:,0, iY] ) )
                    tempMean = -np.arctan2(Y ,X)
                if iY > 0:
                    neiInd = hlp.ravel_multi_index( (iX, iY-1), (nx,ny) )
                    kappa = path[iPath]*self.J
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -XY[:,iX, iY-1] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -XY[:,iX, iY-1] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+XY[:,iX, iY-1] ) )
                    tempMean = -np.arctan2(Y ,X)
                else:
                    neiInd = hlp.ravel_multi_index( (iX, ny-1), (nx,ny) )
                    kappa = path[iPath]*self.J
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -XY[:,iX, ny-1] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -XY[:,iX, ny-1] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+XY[:,iX, ny-1] ) )
                    tempMean = -np.arctan2(Y ,X)
                if iY < ny-1:
                    neiInd = hlp.ravel_multi_index( (iX, iY+1), (nx,ny) )
                    kappa = path[iPath]*self.J
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -XY[:,iX, iY+1] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -XY[:,iX, iY+1] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+XY[:,iX, iY+1] ) )
                    tempMean = -np.arctan2(Y ,X)
                else:
                    neiInd = hlp.ravel_multi_index( (iX,0), (nx,ny) )
                    kappa = path[iPath]*self.J
                    Y = tempDispersion*np.sin( -tempMean ) + kappa*np.sin( -XY[:,iX,0] )
                    X = tempDispersion*np.cos( -tempMean ) + kappa*np.cos( -XY[:,iX,0] )
                    tempDispersion = np.sqrt( tempDispersion**2 + kappa**2 + 2*kappa*tempDispersion*np.cos(-tempMean+XY[:,iX,0] ) )
                    tempMean = -np.arctan2(Y ,X)
                for iSample in range(N):
                    XY[iSample, iX, iY] = hlp.vonmises(tempMean[iSample], tempDispersion[iSample])

            # Calculate weight update log(fl-1/fl)
            for iX in range( nx ):
                for iY in range( ny ):
                    curInd = hlp.ravel_multi_index( (iX, iY), (nx,ny) )
                    pathDiff = path[iPath+1]-path[iPath]
                    if iX > 0:
                        neiInd = hlp.ravel_multi_index( (iX-1, iY), (nx,ny) )
                        logWeights += pathDiff * self.J * np.cos( XY[:,iX, iY] - XY[:,iX-1, iY] )
                    else:
                        neiInd = hlp.ravel_multi_index( (nx-1, iY), (nx,ny) )
                        logWeights += pathDiff * self.J * np.cos( XY[:,iX, iY] - XY[:,nx-1, iY] )                                
                    if iY > 0:
                        neiInd = hlp.ravel_multi_index( (iX, iY-1), (nx,ny) )
                        logWeights += pathDiff * self.J * np.cos( XY[:,iX, iY] - XY[:,iX, iY-1] )
                    else:
                        neiInd = hlp.ravel_multi_index( (iX, ny-1), (nx,ny) )
                        logWeights += pathDiff * self.J * np.cos( XY[:,iX, iY] - XY[:,iX,ny-1] )    
                                
        maxLogWeight = np.max( logWeights )
        weights = np.exp( logWeights - maxLogWeight )
        logZ += maxLogWeight + np.log( np.sum( weights ) ) - np.log( N )
        logZ += nx*ny*np.log(2*np.pi)

        return logZ
                
