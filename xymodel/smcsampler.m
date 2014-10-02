function [X,W,lZ,logW_var] = smcsampler(Jv, Jh, N, beta, par)
% Only works for square models, MxM. Can handle periodic or free boundary
% conditions.
%
% Jv, M x M: vertical interactions, Jv(i,j) is the interaction from X(i,j)
%            to X(i+1,j). The M'th row of Jv containts the boundary
%            interactions between X(M,j) and X(1,j). If omitted, this is
%            assumed to be zero (free boundary condition).
% Jh, M x M: horizontal interactions, Jh(i,j) is the interaction from
%            X(i,j) to X(i,j+1). The M'th column of Jh contains the
%            boundary interactions. If omitted, this is assumed to be zero
%            (free boundary condition).
% N, integer: number of particles
% beta, 1 x (n+1): Annealing sequence, where n is the number of annealing
%                  steps. beta(1) = 0, beta(n+1) = 1
% par.resampling: 1 = multinomial, 2 = stratified, 3 = systematic
%    .ESS_threshold: Value in [0,1], determining when to resample.

M = size(Jv,2);

% For free boundary conditions, assume periodic with zero interaction
if(size(Jv,1) < M)
    Jv = [Jv ; zeros(1,M)];
    Jh = [Jh zeros(M,1)];
end

logW = zeros( N, 1 );
logW_var = zeros(1,length(beta));
nx = M;
ny = M;

% Sample uniformly
%X = 2*pi*rand(nx, ny, N)-pi;
X = zeros(nx,ny,N); % Since we have beta=0 in the first iteration, we will generate independent draws using gibbsgrid
lZ = nx*ny*log(2*pi); % i0(h) should be added if h != 0

% Annealing
for iBeta = 1:length(beta)-1    
    fprintmod(iBeta,round(length(beta)/10));
    
    % Update X
    %X = gibbsgrid(X, beta(iBeta)*Jv, beta(iBeta)*Jh);
    %gibbsgrid_c(X, beta(iBeta)*Jv, beta(iBeta)*Jh);
    U = rand(nx*ny*N*3*3,1); % (N.B. This assumes that we do not need more that 3 runs in the rejection sampling loop on average!)
    gibbsgrid_c_mtlbrnd(X, beta(iBeta)*Jv, beta(iBeta)*Jh, U);    
    
    % Compute weight increment
    betaDiff = beta(iBeta+1) - beta(iBeta);
    logW =  logW + betaDiff*sumdiffX(Jv,Jh,X);        
    logW_var(iBeta+1) = var(logW);

    % Compute ESS and resample if needed
    if(iBeta < length(beta)-1) % Never resample at last iteration
        % Compute ESS
        maxlW = max( logW );
        w = exp( logW - maxlW );
        W = w/sum(w);            
        ESS = 1/(N*sum(W.^2));
        
        if(ESS < par.ESS_threshold)
            ind = resampling(W,par.resampling);
            X = X(:,:,ind);            
            % Update normalizing constant at this iteration
            lZ = lZ + maxlW + log( sum( w ) ) - log( N );
            % Reset weights
            logW = 0;
        end
    end    
end

maxlW = max( logW );
w = exp( logW - maxlW );
lZ = lZ + maxlW + log( sum( w ) ) - log( N );
W = w/sum(w);
end
%--------------------------------------------------------------------------
function S = sumdiffX(Jv,Jh,X)
% Computes sum_{ij} J_{ij}*cos(x_i- x_j) for every page in X
Xdv = X - [X(2:end,:,:) ; X(1,:,:)];
Xdh = X - [X(:,2:end,:)   X(:,1,:)];
S = sum(sum(bsxfun(@times, Jh, cos(Xdh)),1),2) + sum(sum(bsxfun(@times, Jv, cos(Xdv)),1),2);
S = squeeze(S);
end