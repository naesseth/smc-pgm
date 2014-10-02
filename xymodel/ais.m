function [X,W,lZ,logW_var] = ais2(Jv, Jh, K, beta)
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
% K, integer: number of samples
% beta, 1 x (n+1): Annealing sequence, where n is the number of annealing
%                  steps. beta(1) = 0, beta(n+1) = 1

M = size(Jv,2);

% For free boundary conditions, assume periodic with zero interaction
if(size(Jv,1) < M)
    Jv = [Jv ; zeros(1,M)];
    Jh = [Jh zeros(M,1)];
end

logW = zeros( K, 1 );
logW_var = zeros(1,length(beta));
nx = M;
ny = M;

% Sample uniformly
%X = 2*pi*rand(nx, ny, K)-pi;
X = zeros(nx,ny,K); % Since we have beta=0 in the first iteration, we will generate independent draws using gibbsgrid
% Annealing
for iBeta = 1:length(beta)-1
    fprintmod(iBeta,round(length(beta)/10));
    
    % Update X
    %X = gibbsgrid(X, beta(iBeta)*Jv, beta(iBeta)*Jh);
    %gibbsgrid_c(X, beta(iBeta)*Jv, beta(iBeta)*Jh);
    U = rand(nx*ny*K*3*3,1); % (N.B. This assumes that we do not need more that 3 runs in the rejection sampling loop on average!)
    gibbsgrid_c_mtlbrnd(X, beta(iBeta)*Jv, beta(iBeta)*Jh, U);    
    
    % Compute weight increment
    betaDiff = beta(iBeta+1) - beta(iBeta);
    logW =  logW + betaDiff*sumdiffX(Jv,Jh,X);
    logW_var(iBeta+1) = var(logW);
end
maxlW = max( logW );
w = exp( logW - maxlW );
lZ = maxlW + log( sum( w ) ) - log( K );
lZ = lZ + nx*ny*log(2*pi); % i0(h) should be added if h != 0
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