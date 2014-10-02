addpath('./hlp');

%% Setup
% - Base settings
n = 6;
numMC = 10;

par.resampling = 3;
par.ESS_threshold = 0.5;

% ---- Computational budget / # of particles and annealing steps ----
N_ann = [10 100];
numBeta_ann = [10000 1000];
Jvec = [0.5 1.1 1.7];

% - Model, periodic and random interactions
M = 2^n;
% -- Constant, free
% Jv = ones(M-1,M); % Vertical interactions
% Jh = ones(M,M-1); % Horizontal interactions
% -- Constant periodic,
Jv = ones(M);
Jh = ones(M);
% -- Random, periodic
% Jv = 2*rand(M)-1;
% Jh = 2*rand(M)-1;

numMethods = 2*2*length(N_ann)*length(Jvec); % 2 annealing schedules, 2 methods

timeVec = zeros(numMC,numMethods);
essVec =zeros(numMC,numMethods);
lZ = zeros(numMC,numMethods);
xhat = zeros(M,M,numMC,numMethods);


%% Run algorithms
for(jj = 1:numMC)
    %%%%%% Ugly way of keeping track of which setting we use at the moment
    methodId = 0;
    %%%%%%

    for(tt = 1:length(Jvec))
        Jv_now = Jvec(tt)*Jv;
        Jh_now = Jvec(tt)*Jh;
        
        for(kk = 1:length(N_ann))
            % Current settings
            N = N_ann(kk);
            numBeta = numBeta_ann(kk);
            
            fprintf('MC(%i), beta(%2.2f), N(%i), numBeta(%i)\n',jj,Jvec(tt),N,numBeta);
            
            for(scheduleId = 1:2) % Geometric or linear
                if(scheduleId == 1) % Geometric                    
                    if(numBeta == 10000)
                        beta = zeros(10000,1);
                        beta(2:500) = logspace(-8,-6,499);
                        beta(501:5000) = logspace(-5.99,-1.3,4500);
                        beta(5001:end) = logspace(-1.29,0,5000);
                    elseif(numBeta == 1000)
                        beta = zeros(1000,1);
                        beta(2:50) = logspace(-8,-6,49);
                        beta(51:500) = logspace(-5.99,-1.3,450);
                        beta(501:end) = logspace(-1.29,0,500);
                    end         
                else % Linear
                    beta = linspace(0,1,numBeta);
                end
                
                % ---AIS
                methodId = methodId + 1; % lazy way of computing the methodId
                tic;
                [x_tmp,w_tmp,lZ(jj,methodId)] = ais(Jv_now,Jh_now,N,beta);
                timeVec(jj,methodId) = toc;
                essVec(jj,methodId) = 1/sum(w_tmp(:).^2);
                xhat(:,:,jj,methodId) = sum(bsxfun(@times, x_tmp, reshape(w_tmp(:),[1 1 N])),3);
                clear x_tmp w_tmp;
                
                % ---SMC sampler (referred to as ASIR in the paper)
                methodId = methodId + 1; % lazy way of computing the methodId
                tic;
                [x_tmp,w_tmp,lZ(jj,methodId)] = smcsampler(Jv_now,Jh_now,N,beta,par);
                timeVec(jj,methodId) = toc;
                essVec(jj,methodId) = 1/sum(w_tmp(:).^2);
                xhat(:,:,jj,methodId) = sum(bsxfun(@times, x_tmp, reshape(w_tmp(:),[1 1 N])),3);
                clear x_tmp w_tmp;
            end
        end
    end

    save('/data/lindsten/dacsmc/XYmodel/pgmM64');
end

