/*
 * gibbsgrid.c - MEX routine for Gibbs-sampling the XY model
 *
 * Inputs: 
 *
 * X, nx x ny x N: Current state of the Markov chain
 * Jv, nx x ny: vertical interactions, Jv(i,j) is the interaction from X(i,j)
 *              to X(i+1,j). The M'th row of Jv containts the boundary
 *              interactions between X(M,j) and X(1,j).
 * Jh, nx x ny: horizontal interactions, Jh(i,j) is the interaction from
 *              X(i,j) to X(i,j+1). The M'th column of Jh contains the
 *              boundary interactions.  
 * 
 * Outputs:
 *
 * void, the method operates on the input argument.
 %
 * The calling syntax is:
 *
 *		gibbsgrid(Xnow, Jv, Jh)
 *
 * For use within AIS/SMC samplers, the function supports input argument X
 * to be of dimension M x M x N, in which case N updates are made in
 * parallel.
 *
 * This is a MEX-file for MATLAB.
 */

#include "mex.h"
#include "math.h"
#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

/* uniform [0,1] pseudo-random number generator */
double unirand(mwSize precision) {
    double result = 0.0;
    mwSize i;
    for(i=0; i<precision; ++i) {
        result = (result + rand()) / ((double)RAND_MAX + 1.0);
    }
    return result;
}
/* von-mises pseude-random number generator */
double vmrnd(double mean, double k)
{
    const mwSize precision = 3; /* 15 bit per precision value*/   
    double result = 0.0;    
    double a = 1.0 + sqrt(1 + 4.0 * (k * k));
    double b = (a - sqrt(2.0 * a))/(2.0 * k);
    double r = (1.0 + b * b)/(2.0 * b);
    double U1, U2, U3, z, f, c, sign;

    if(k < 0.05) { /* Use uniform proposal */
        while(1) {
            z = 2*M_PI*unirand(precision)-M_PI;
            U1 = unirand(precision);
            
            if(exp(k*(cos(z)-1))-U1 > 0.0) {
                result = z+mean;
                break;
            }
        }
    }
    else { /* Use Cauchy proposal */
        while (1) {
            U1 = unirand(precision);
            z = cos(M_PI * U1);
            f = (1.0 + r * z)/(r + z);
            c = k * (r - f);
            U2 = unirand(precision);
                
            if (c * (2.0 - c) - U2 > 0.0) {
                U3 = unirand(precision);
                sign = 0.0;
                if (U3 - 0.5 < 0.0)
                    sign = -1.0;
                if (U3 - 0.5 > 0.0)
                    sign = 1.0;
                result = sign * acos(f) + mean;
                break;
            }
            else {
                if(log(c/U2) + 1.0 - c >= 0.0) {
                    U3 = unirand(precision);
                    sign = 0.0;
                    if (U3 - 0.5 < 0.0)
                        sign = -1.0;
                    if (U3 - 0.5 > 0.0)
                        sign = 1.0;
                    result = sign * acos(f) + mean;
                    break;
                }
            }
        }
    }

    /* Make sure that we get something in the correct range */
    while (result >= M_PI)
        result -= 2.0 * M_PI;
    while (result <= -M_PI)
        result += 2.0 * M_PI;

    return result;
}

/* update a single site */
void updateSite(double *X, double *Jv, double *Jh, const mwSize nx, const mwSize ny, const mwSize numN,
        const mwSize ii, const mwSize iX, const mwSize iY)
{
    double center, dispersion, alpha, beta;
    const long offset=nx*ny*ii;
    /* above */
    if(iX==0) {
        alpha = Jv[nx*iY + nx-1]*cos(X[offset + nx*iY + nx-1]);
        beta = Jv[nx*iY + nx-1]*sin(X[offset + nx*iY + nx-1]);
    }
    else {
        alpha = Jv[nx*iY + iX-1]*cos(X[offset + nx*iY + iX-1]);
        beta = Jv[nx*iY + iX-1]*sin(X[offset + nx*iY + iX-1]);
    }
    /* below */
    if(nx > 1) { /* only count the interaction once for a 1d model */
        if(iX==nx-1) {
            alpha += Jv[nx*iY + iX]*cos(X[offset + nx*iY]);
            beta += Jv[nx*iY + iX]*sin(X[offset + nx*iY]);
        }
        else {
            alpha += Jv[nx*iY + iX]*cos(X[offset + nx*iY + iX+1]);
            beta += Jv[nx*iY + iX]*sin(X[offset + nx*iY + iX+1]);
        }
    }
    /* left */
    if(iY==0) {
        alpha += Jh[nx*(ny-1) + iX]*cos(X[offset + nx*(ny-1) + iX]);
        beta += Jh[nx*(ny-1) + iX]*sin(X[offset + nx*(ny-1) + iX]);
    }
    else {
        alpha += Jh[nx*(iY-1) + iX]*cos(X[offset + nx*(iY-1) + iX]);
        beta += Jh[nx*(iY-1) + iX]*sin(X[offset + nx*(iY-1) + iX]);
    }
    /* right */
    if(ny > 1) { /* only count the interaction once for a 1d model */
        if(iY==ny-1) {
            alpha += Jh[nx*iY + iX]*cos(X[offset + iX]);
            beta += Jh[nx*iY + iX]*sin(X[offset + iX]);
        }
        else {
            alpha += Jh[nx*iY + iX]*cos(X[offset + nx*(iY+1) + iX]);
            beta += Jh[nx*iY + iX]*sin(X[offset + nx*(iY+1) + iX]);
        }
    }
    
    /* Compute the von-Mises conditional */
    dispersion = sqrt(alpha*alpha + beta*beta);
    center = atan2(beta, alpha);
    X[offset + nx*iY + iX] = vmrnd(center, dispersion);    
}
    
/* main loop */
void gibbsgrid(double *X, double *Jv, double *Jh, const mwSize ndim, const mwSize* dims)
{
    mwSize numN;
    const mwSize nx = dims[0];
    const mwSize ny = dims[1];
    mwSize ii, iX, iY;
    
    /* extract dimensions */
    numN = (ndim == 2 ? 1 : dims[2]);
    
    /* deterministic scan gibbs */
    for(ii=0; ii<numN; ++ii) {
        for(iY=0; iY<ny; ++iY) {            
            for(iX=0; iX<nx; ++iX) {
                updateSite(X,Jv,Jh,nx,ny,numN,ii,iX,iY);
            }
        }
    }       
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* variable declarations here */
    double *X;         /* nx x ny x N data matrix */
    double *Jv, *Jh;   /* nx x ny input matrices */
    const mwSize *dims;
    mwSize ndim;
    
    /* check input and output arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MyToolbox:gibbsgrid:nrhs", "Three inputs required.");
    }
    if(nlhs>0) {
        mexErrMsgIdAndTxt("MyToolbox:gibbsgrid:nlhs", "Too many output arguments.");
    }
    /* make sure the input arguments are double matrices */
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])
        || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])
        || !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2])) {
        mexErrMsgIdAndTxt("MyToolbox:gibbsgrid:input", "Input matrices must be a scalar.");
    }
    
    /* get the input dimensions */
    ndim = mxGetNumberOfDimensions(prhs[0]);
    dims = mxGetDimensions(prhs[0]);
    
    /* should check Jv and Jh here as well */
    
    /* check that we have a lattice */
    if( ndim < 2 || ndim > 3 ) {
        mexErrMsgIdAndTxt("MyToolbox:gibbsgrid:input", "Wrong dimension on input arrays.");
    }
            
    /* Get pointers to the input arguments */
    X = mxGetPr(prhs[0]);
    Jv = mxGetPr(prhs[1]);
    Jh = mxGetPr(prhs[2]);
    
    /* Set the output pointer to be the same as the input array */
    /* plhs[0] = prhs[0]; */

    /* mexPrintf("%d.\n",RAND_MAX); */ 
    /* mexPrintf("%d.\n",sizeof(long)); */

    /* call the computational routine */
    gibbsgrid(X,Jv,Jh,ndim,dims);    
}
