% Estimation and Detection Theory - AUTh 2024
% Topic 1: Exponential Parameter Estimation
% Part 2: Maximum Likelihood Estimator (MLE) Simulation
% Authors: Dimitrios Ioannidis, Anastasios Theocharis Sotiropoulos
%%
clear; close all; clc;

lambda = 2;              % True parameter value (lambda = 2)
N = 10;                  % Number of observations per experiment
experiments = 5000;      % Number of Monte Carlo experiments

%% Maximum Likelihood Estimator (MLE) Histogram
mean_dist = 1 / lambda;  % Mean of the exponential distribution (1/lambda)

lambda_estim = zeros(1, experiments);
has_stats = exist('exprnd', 'file');

for i = 1:experiments
    if has_stats
        x_simul = exprnd(mean_dist, 1, N);
    else
        x_simul = -mean_dist * log(rand(1, N));
    end
    lambda_estim(i) = 1 / mean(x_simul);
end

% Plot the histogram
fig = figure('Name', sprintf('Histogram for lambda = %d', lambda), 'NumberTitle', 'off', 'Position', [100 50 800 600]);
histogram(lambda_estim, 'BinWidth', 0.10, 'FaceColor', [0.85 0.33 0.1]);
xlabel('Estimated \lambda (MLE)', 'FontSize', 14);
ylabel('Frequency', 'FontSize', 14);
title(sprintf('Histogram of MLE (\\lambda = %d, N = %d, %d Experiments)', lambda, N, experiments), 'FontSize', 15);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
grid on;

mean_MLE = mean(lambda_estim);
var_MLE = var(lambda_estim);
legend(sprintf('Mean: %.3f, Variance: %.3f', mean_MLE, var_MLE), 'Location', 'northeast');

% Export figure to assets
output_dir = fullfile('..', 'assets', '01-exponential-parameter-estimation');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
exportgraphics(fig, fullfile(output_dir, sprintf('histogram_mle_lambda_%d.png', lambda)));
