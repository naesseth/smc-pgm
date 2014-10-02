function lZ = lrs2(w, phi, m, R)
% words,        w:   1 x n
% topics,       phi: T x V
% topic_prior,  m:   T x 1

alpha = 1;
n = length(w);
T = length(m);
burnin = ceil(R/10);

z = zeros(R,n);

% Initialize
gm = phi(:,w(1)).*m;
lZ = log(sum(gm)); % log p(y_1)
z = discreternd(R, gm/sum(gm));

% Count occurences of different topics in z
tau = zeros(R,T);
ti = sub2ind([R T], (1:R)', z);
tau(ti) = 1;

for(k = 2:n)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute log p(y_k | y_{1:k-1}) based on Gibbs samples from _previous_
    % iteration z_{1:k-1}    
%     gm = ( bsxfun(@times, phi(:,w(k)), bsxfun(@plus, tau', alpha*m))/(k-1+alpha) )';
%     nu = sum(gm,2);
%     lZ = lZ + log(sum(nu))-log(R);
    
    % Alternative way, with burnin when computing the estimates
    gm = ( bsxfun(@times, phi(:,w(k)), bsxfun(@plus, tau(burnin+1:R,:)', alpha*m))/(k-1+alpha) )';
    nu = sum(gm,2);
    lZ = lZ + log(sum(nu))-log(R-burnin);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Run a Gibbs sampler for p(z_{1:k} | y_{1:k}) - this will be used in
    % the next iteration
    [z,tau] = gibbs_updates(z(end,:), tau(end,:), gm(end,:), alpha, w, phi, m, R);
    
end

end
%--------------------------------------------------------------------------
function [z,tau] = gibbs_updates(z_init, tau_init, gm, alpha, w, phi, m, R)
% z_init : 1 x k-1, last iteration from previous sampler
% tau_init : 1 x T, corresponding suff stat
% gm : 1 x T, (unnormalized) distribution over z_k | z_{1:k-1}, y_{1:k} (used to compute
%             the likelihood in the previous iteration)
%
% z: R x k
% tau : R x T

k = size(z_init,2)+1; % Expanding one step
T = size(tau_init,2);
z = zeros(R,k);
tau = zeros(R,T);

% Augment the initial state
z(1,1:k-1) = z_init;
tau(1,:) = tau_init;
z(1,k) = discreternd(1, gm/sum(gm));
tau(1,z(1,k)) = tau(1,z(1,k))+1;

for(r = 2:R) % Gibbs iterations
    z_now = z(r-1,:);
    tau_now = tau(r-1,:);
    
    for(kprime = 1:k) % Forward sequential, but start with the last entry since this is not initialized
        % Remove z_kprime
        tau_now(z_now(kprime)) = tau_now(z_now(kprime)) - 1;
        gm = ((phi(:,w(kprime)).*(tau_now' + alpha*m))/(k-1+alpha))';
        z_now(kprime) = discreternd(1, gm/sum(gm));
        tau_now(z_now(kprime)) = tau_now(z_now(kprime))+1;
    end
    z(r,:) = z_now;
    tau(r,:) = tau_now;
end


end