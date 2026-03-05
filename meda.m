%% Load data
close all
clear
clc
load data.mat
size(X_fft)
%% varPca Plots
% varPca(X_fft, "PCs", 1:10, "Preprocessing", 1, "PlotCkf", false);
% varPca(X_raw, "PCs", 1:10, "Preprocessing", 0, "PlotCkf", false);
% varPca(X_sample_scaled, "PCs", 1:10, "Preprocessing", 0, "PlotCkf", false);
% varPca(X_block_scaled, "PCs", 1:10, "Preprocessing", 0, "PlotCkf", false);
% varPca(X_norm, "PCs", 1:10, "Preprocessing", 0, "PlotCkf", false);

%% fit PCA model
X = X_raw;

X = preprocess2D(X, 'Preprocessing', 1);
% varPca(X, "PCs", 1:10, "Preprocessing", 0, "PlotCkf", false);

model.lvs=1:2;
model = pcaEig(X,'PCs',model.lvs);
model.var = trace(X'*X);

% Labels and classes
freq_label = repmat(freqs, 1, length(times));
freq_label_string = string(repmat(freqs, 1, length(times)))+"Hz";
time_label = repmat(times, 1, length(freqs));
%% PCA Visualization
for i=1:4
    class = F_data(:,i);
    scores(model, "ObsLabel", class, "ObsClass", class, "Color", "okabeIto");
end
%%
loadings(model, "VarsLabel", time_label, "VarsClass", time_label, "Color", "parula");
%%
loadings(model, "VarsLabel", freq_label, "VarsClass", freq_label, "Color", "parula");

%% ASCA
% clc
[T, model] = parglm(X, F_data, 'Preprocessing', 0, 'Model', 'interaction');
T.Source(1:4) = {'Time', 'Treatment', 'Sex', 'Order'};
T

%%
ascao = asca(model);
factor_id = 1;
ascao.factors{factor_id}.lvs=1:1;
scores(ascao.factors{factor_id}, 'ObsClass', F_data(:, factor_id));

%%
interaction_id = 1;
ascao.interactions{interaction_id}.lvs=1:2;
scores(ascao.interactions{interaction_id}, 'ObsClass', F_data(:, interaction_id));
