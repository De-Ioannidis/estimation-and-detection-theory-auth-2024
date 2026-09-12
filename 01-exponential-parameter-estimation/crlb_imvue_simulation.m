% Estimation and Detection Theory - AUTh 2024
% Topic 1: Exponential Parameter Estimation
% Part 1: Cramer-Rao Lower Bound (CRLB) & Ideal MVUE Simulation
% Authors: Dimitrios Ioannidis, Anastasios Theocharis Sotiropoulos
%%
clear; close all; clc;

lambda = 2;              % True parameter value (lambda = 2)
N = 10;                  % Number of observations per experiment
experiments = 5000;      % Number of Monte Carlo experiments

%% MVU Estimator Histogram
mean_dist = 1 / lambda;  % Mean of the exponential distribution (1/lambda)

% CRLB for Exponential distribution: I(lambda) = N / lambda^2 => CRLB = lambda^2 / N
CRLB = lambda^2 / N;

if exist('normrnd', 'file')
    lambda_estim = normrnd(lambda, sqrt(CRLB), [1, experiments]);
else
    lambda_estim = lambda + sqrt(CRLB) * randn(1, experiments);
end

% Plot the histogram
fig = figure('Name', sprintf('Histogram for lambda = %d', lambda), 'NumberTitle', 'off', 'Position', [100 50 800 600]);
histogram(lambda_estim, 'BinWidth', 0.10, 'FaceColor', [0.2 0.4 0.8]);
xlabel('Estimated \lambda (IMVUE)', 'FontSize', 14);
ylabel('Frequency', 'FontSize', 14);
title(sprintf('Histogram of IMVUE (\\lambda = %d, N = %d, %d Experiments)', lambda, N, experiments), 'FontSize', 15);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
grid on;

mean_IMVUE = mean(lambda_estim);
var_IMVUE = var(lambda_estim);
legend(sprintf('Mean: %.3f, Variance: %.3f (CRLB: %.3f)', mean_IMVUE, var_IMVUE, CRLB), 'Location', 'northeast');

% Export figure to assets
output_dir = fullfile('..', 'assets', '01-exponential-parameter-estimation');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
exportgraphics(fig, fullfile(output_dir, sprintf('histogram_imvue_lambda_%d.png', lambda)));
