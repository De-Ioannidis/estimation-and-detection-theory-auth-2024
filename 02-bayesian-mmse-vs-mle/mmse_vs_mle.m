% Estimation and Detection Theory - AUTh 2024
% Topic 2: Classical Linear Model & Bayesian Estimation
% Part 3: MSE Convergence Comparison (Bayesian MMSE vs Classical MLE)
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

has_stats = exist('normrnd', 'file');

%% MMSE Estimator: Bayesian Gaussian-Gaussian Model
MMSE = @(x_n) 2 / (16 + length(x_n)) * sum(x_n) - 4 * length(x_n) / (16 + length(x_n)) + 4;

mse_observations_mmse = zeros(1, max_observations);
for i = 1:max_observations
    mse_exp = zeros(1, experiments);
    for j = 1:experiments
        if has_stats
            th_n = normrnd(mu_th, sigma_th);
            w_n = normrnd(mu_w, sigma_w, 1, i);
        else
            th_n = mu_th + sigma_th * randn();
            w_n = mu_w + sigma_w * randn(1, i);
        end
        x_n = h * th_n + w_n;
        
        th_estim_MMSE = MMSE(x_n);
        mse_exp(j) = (th_estim_MMSE - th_n)^2;
    end
    mse_observations_mmse(i) = mean(mse_exp);
end

%% MLE Estimator: Deterministic Unknown Parameter (theta = 4)
MLE = @(x_n) 2 / length(x_n) * sum(x_n);

mse_observations_mle = zeros(1, max_observations);
th_deterministic = 4;

for i = 1:max_observations
    mse_exp = zeros(1, experiments);
    for j = 1:experiments
        if has_stats
            w_n = normrnd(mu_w, sigma_w, 1, i);
        else
            w_n = mu_w + sigma_w * randn(1, i);
        end
        x_n = h * th_deterministic + w_n;
        
        th_estim_MLE = MLE(x_n);
        mse_exp(j) = (th_estim_MLE - th_deterministic)^2;
    end
    mse_observations_mle(i) = mean(mse_exp);
end

%% Plotting & Asset Generation
output_dir = fullfile('..', 'assets', '02-bayesian-mmse-vs-mle');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 1. MMSE Convergence
fig1 = figure('Name', 'MSE Convergence (MMSE)', 'NumberTitle', 'off', 'Position', [100 50 800 550]);
plot(1:max_observations, mse_observations_mmse, 'LineWidth', 2, 'Color', [0.15 0.35 0.8]);
hold on;
line([1 max_observations], [0.25 0.25], 'Color', [0.4 0.4 0.4], 'LineStyle', '--', 'LineWidth', 1.5);
grid on;
xlabel('Number of Observations N per Experiment', 'FontSize', 13);
ylabel('Mean Squared Error (MSE)', 'FontSize', 13);
title(sprintf('MMSE Convergence vs Observations N (%d Experiments, \\theta \\sim N(4, 1))', experiments), 'FontSize', 14);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
legend('Simulated MMSE', 'Asymptotic Floor (y = 0.25)', 'Location', 'northeast');
exportgraphics(fig1, fullfile(output_dir, 'mse_convergence_mmse.png'));

% 2. MLE Convergence
fig2 = figure('Name', 'MSE Convergence (MLE)', 'NumberTitle', 'off', 'Position', [100 50 800 550]);
plot(1:max_observations, mse_observations_mle, 'LineWidth', 2, 'Color', [0.85 0.33 0.1]);
hold on;
line([1 max_observations], [0.33 0.33], 'Color', [0.4 0.4 0.4], 'LineStyle', '--', 'LineWidth', 1.5);
grid on;
xlabel('Number of Observations N per Experiment', 'FontSize', 13);
ylabel('Mean Squared Error (MSE)', 'FontSize', 13);
title(sprintf('MLE Convergence vs Observations N (%d Experiments, Deterministic \\theta = 4)', experiments), 'FontSize', 14);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
legend('Simulated MLE', 'Reference Benchmark (y = 0.33)', 'Location', 'northeast');
exportgraphics(fig2, fullfile(output_dir, 'mse_convergence_mle.png'));

% 3. Dual Comparison Plot
fig3 = figure('Name', 'MSE Comparison (MMSE vs MLE)', 'NumberTitle', 'off', 'Position', [100 50 850 550]);
plot(1:max_observations, mse_observations_mmse, 'LineWidth', 2, 'Color', [0.15 0.35 0.8]);
hold on;
plot(1:max_observations, mse_observations_mle, 'LineWidth', 2, 'Color', [0.85 0.33 0.1]);
grid on;
xlabel('Number of Observations N per Experiment', 'FontSize', 13);
ylabel('Mean Squared Error (MSE)', 'FontSize', 13);
title(sprintf('Convergence Comparison: Bayesian MMSE vs Deterministic MLE (%d Experiments)', experiments), 'FontSize', 14);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
legend('Bayesian MMSE (exploits prior)', 'Classical MLE (unbiased)', 'Location', 'northeast');
exportgraphics(fig3, fullfile(output_dir, 'mse_convergence_comparison.png'));
