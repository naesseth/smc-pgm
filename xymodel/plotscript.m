load pgmM64;

%% All
fprintf('Completed %i/%i runs.\n',jj,numMC');
%%
inds = cell(2,3);
for(tt = 1:length(Jvec))
    inds{1,tt} = (numMethods/3)*(tt-1) + (1:2:(numMethods/3)); % AIS
    inds{2,tt} = (numMethods/3)*(tt-1) + (2:2:(numMethods/3)); % SMC
end

numMethodTypes = 2;
titles = {'AIS','SMC'};
colors = {'r','b'};

%% ESS plot

for(tt = 1:length(Jvec))
    figure(tt);
    for(methodType = 1:numMethodTypes)
        subplot(2,1,methodType);
        plot(1:length(inds{methodType,tt}), essVec(1:jj,inds{methodType,tt}));
        title(titles{methodType});
    end
end

%% Z-plot
close all;
H = zeros(1,numMethodTypes);

for(tt = 1:length(Jvec))
    figure(tt);
    for(methodType = 1:numMethodTypes)
        Htmp=plot(inds{methodType,tt}, lZ(1:jj,inds{methodType,tt}),'linestyle','none','Marker','o',...
            'MarkerSize',12,'MarkerFaceColor',colors{methodType},'MarkerEdgeColor','k','linewidth',2);
        H(methodType) = Htmp(1);
        hold on;
        
        for(method = inds{methodType,tt})
            Htmp = text(method,1e4,num2str(0.1*round(10*mean(essVec(1:jj,method)))));
            set(Htmp,'Rotation',90)
        end
    end

%     ylim([9600 10100]);
%     xlim([0 20]);
    legend(H,titles,'Location','SouthWest')
    title(sprintf('log(Z): J=%2.2f, N = %i, %i (reverse for #beta), geometric/linear schedule',Jvec(tt),N_ann))
    hold off;
end
%% MSE

xmse = squeeze(mean(xhat(:,:,1:jj,:).^2,3));
range = 10*log10([min(xmse(:)), max(xmse(:))]);
figure;
for(mtd = 1:numMethods)
    subplot(3,8,mtd);
    imagesc(10*log10(xmse(:,:,mtd)), range); colorbar;
end
