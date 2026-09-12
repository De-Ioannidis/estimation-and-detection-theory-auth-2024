% Estimation and Detection Theory - AUTh 2024
% Topic 2: Classical Linear Model & Bayesian Estimation
% Part 4: Robustness Analysis Under Prior Misspecification
% Authors: Dimitrios Ioannidis, Anastasios Theocharis Sotiropoulos
%%
clear; close all; clc;

experiments = 500;       % Monte Carlo realizations
max_observations = 50;   % Up to N = 50 observations

mu_w = 0;
sigma_w = 2;
mu_th = 4;
sigma_th = 1;
h = 0.5;

has_norm = exist('normrnd', 'file');
has_unif = exist('unifrnd', 'file');
has_exp = exist('exprnd', 'file');

% Original MMSE estimator derived under Gaussian prior N(4, 1):
MMSE = @(x_n) 2 / (16 + length(x_n)) * sum(x_n) - 4 * length(x_n) / (16 + length(x_n)) + 4;

%% 1. Inaccurate Prior: Uniform Distribution U[a, b]
a = mu_th - sqrt(3) * sigma_th; % 4 - sqrt(3) ~ 2.268
b = mu_th + sqrt(3) * sigma_th; % 4 + sqrt(3) ~ 5.732

mse_observations_unif = zeros(1, max_observations);
for i = 1:max_observations
    mse_exp = zeros(1, experiments);
    for j = 1:experiments
        if has_norm
            w_n = normrnd(mu_w, sigma_w, 1, i);
        else
            w_n = mu_w + sigma_w * randn(1, i);
        end
        if has_unif
            th_n = unifrnd(a, b);
        else
            th_n = a + (b - a) * rand();
        end
        x_n = h * th_n + w_n;
        
        th_estim = MMSE(x_n);
        mse_exp(j) = (th_estim - th_n)^2;
    end
    mse_observations_unif(i) = mean(mse_exp);
end

%% 2. Inaccurate Prior: Exponential Distribution Exp(lambda)
lambda_prior = 1 / mu_th; % lambda = 0.25 (variance = 16)

mse_observations_exp = zeros(1, max_observations);
for i = 1:max_observations
    mse_exp = zeros(1, experiments);
    for j = 1:experiments
        if has_norm
            w_n = normrnd(mu_w, sigma_w, 1, i);
        else
            w_n = mu_w + sigma_w * randn(1, i);
        end
        if has_exp
            th_n = exprnd(1 / lambda_prior);
        else
            th_n = -(1 / lambda_prior) * log(rand());
        end
        x_n = h * th_n + w_n;
        
        th_estim = MMSE(x_n);
        mse_exp(j) = (th_estim - th_n)^2;
    end
    mse_observations_exp(i) = mean(mse_exp);
end

%% Plotting & Asset Generation
output_dir = fullfile('..', 'assets', '02-bayesian-mmse-vs-mle');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Plot 1: Uniform Misspecification
fig1 = figure('Name', 'MMSE with Uniform Prior', 'NumberTitle', 'off', 'Position', [100 50 800 550]);
plot(1:max_observations, mse_observations_unif, 'LineWidth', 2, 'Color', [0.15 0.35 0.8]);
hold on;
line([1 max_observations], [0.25 0.25], 'Color', [0.4 0.4 0.4], 'LineStyle', '--', 'LineWidth', 1.5);
grid on;
xlabel('Number of Observations N per Experiment', 'FontSize', 13);
ylabel('Mean Squared Error (MSE)', 'FontSize', 13);
title(sprintf('MMSE Convergence with Misspecified Prior (Uniform U[%.2f, %.2f])', a, b), 'FontSize', 14);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
legend('Simulated MSE (Uniform Ground Truth)', 'Reference Floor (y = 0.25)', 'Location', 'northeast');
exportgraphics(fig1, fullfile(output_dir, 'mse_convergence_uniform_prior.png'));

% Plot 2: Exponential Misspecification
fig2 = figure('Name', 'MMSE with Exponential Prior', 'NumberTitle', 'off', 'Position', [100 50 800 550]);
plot(1:max_observations, mse_observations_exp, 'LineWidth', 2, 'Color', [0.85 0.33 0.1]);
hold on;
line([1 max_observations], [1.05 1.05], 'Color', [0.4 0.4 0.4], 'LineStyle', '--', 'LineWidth', 1.5);
grid on;
xlabel('Number of Observations N per Experiment', 'FontSize', 13);
ylabel('Mean Squared Error (MSE)', 'FontSize', 13);
title(sprintf('MMSE Convergence with Misspecified Prior (Exponential \\lambda = %.2f, \\sigma^2 = 16)', lambda_prior), 'FontSize', 14);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
legend('Simulated MSE (Exponential Ground Truth)', 'Reference Threshold (y = 1.05)', 'Location', 'northeast');
exportgraphics(fig2, fullfile(output_dir, 'mse_convergence_exponential_prior.png'));
