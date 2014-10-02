function lZ = fapf(w, phi, m, Np)
% words,        w:   1 x n
% topics,       phi: T x V
% topic_prior,  m:   T x 1

alpha = sum(m);
n = length(w);
T = length(m);

z = zeros(Np,n);

% Initialize
gm = phi(:,w(1)).*m;
lZ = log(sum(gm));
z(:,1) = discreternd(Np, gm/sum(gm));

tau = zeros(Np,T);
ti = sub2ind([Np T], (1:Np)', z(:,1));
tau(ti) = 1;

for k = 2:n
    gm = ( bsxfun(@times, phi(:,w(k)), bsxfun(@plus, tau', alpha*m))/(k-1+alpha) )';
    nu = sum(gm,2);
    lZ = lZ + log(sum(nu))-log(Np);
    % Resample (not ideal to resample at every time step, but ok..)
    ind = resampling(nu/sum(nu),3);
    tau = tau(ind,:);
    gm = gm(ind,:);
    nu = nu(ind);
    % Propagate (this can be done in a better way!!)    
    gm = bsxfun(@rdivide, gm, nu);
    z_tmp = repmat(rand(Np,1),[1 T]) < cumsum(gm,2);
    z(:,k) = T+1-sum(z_tmp,2);
    ti = sub2ind([Np T], (1:Np)', z(:,k));
    tau(ti) = tau(ti)+1;
end