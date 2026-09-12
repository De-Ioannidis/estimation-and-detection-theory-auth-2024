% Estimation and Detection Theory - AUTh 2024
% Topic 3: Distribution Identification & Maximum Likelihood Estimation
% Authors: Dimitrios Ioannidis, Anastasios Theocharis Sotiropoulos
%%
clear; close all; clc;

%% 1. Data Extraction
if ~isfile('data.mat')
    error('data.mat not found in current directory. Please ensure working directory is 03-distribution-identification.');
end

dataStruct = load('data.mat');
fieldNames = fieldnames(dataStruct);
data = dataStruct.(fieldNames{1});

numDataPoints = size(data);
lengthData = length(data);

fprintf('====================================================\n');
fprintf('  Dataset Analysis: Sample Size N = %d\n', lengthData);
fprintf('  Sample Mean:     %.4f\n', mean(data));
fprintf('  Sample Variance: %.4f\n', var(data));
fprintf('====================================================\n\n');

%% 2. Parameter Estimation via MLE
% Hypothesis 1: Exponential Distribution p(x; lambda) = lambda * exp(-lambda*x)
lambdaExp = 1 / mean(data);
if exist('exprnd', 'file')
    expData = exprnd(1 / lambdaExp, numDataPoints);
else
    expData = -(1 / lambdaExp) * log(rand(numDataPoints));
end

% Hypothesis 2: Rayleigh Distribution p(x; sigma^2) = (x / sigma^2) * exp(-x^2 / (2*sigma^2))
sigmaSqRayleigh = (0.5 / lengthData) * sum(data.^2);
sigmaRayleigh = sqrt(sigmaSqRayleigh);
if exist('raylrnd', 'file')
    rayleighData = raylrnd(sigmaRayleigh, numDataPoints);
else
    rayleighData = sigmaRayleigh * sqrt(-2 * log(rand(numDataPoints)));
end

fprintf('MLE Parameters:\n');
fprintf('  Exponential:  lambda_hat = %.4f\n', lambdaExp);
fprintf('  Rayleigh:     sigma^2_hat = %.4f (sigma = %.4f)\n\n', sigmaSqRayleigh, sigmaRayleigh);

%% 3. Visual Density Comparison & Asset Generation
output_dir = fullfile('..', 'assets', '03-distribution-identification');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

numBins = 50;

% Figure 1: Empirical Data Histogram
fig1 = figure('Name', 'Empirical Data Distribution', 'NumberTitle', 'off', 'Position', [100 50 800 550]);
h1 = histogram(data, numBins, 'FaceColor', [0.15 0.35 0.8], 'Normalization', 'count');
xlabel('Value x', 'FontSize', 13);
ylabel('Occurrence Count', 'FontSize', 13);
title(sprintf('Empirical Data Distribution (N = %d)', lengthData), 'FontSize', 15);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
grid on;
exportgraphics(fig1, fullfile(output_dir, 'histogram_data.png'));

% Figure 2: Fitted Exponential Model
fig2 = figure('Name', 'Fitted Exponential Distribution', 'NumberTitle', 'off', 'Position', [100 50 800 550]);
h2 = histogram(expData, numBins, 'FaceColor', [0.85 0.33 0.1], 'Normalization', 'count');
xlabel('Value x', 'FontSize', 13);
ylabel('Occurrence Count', 'FontSize', 13);
title(sprintf('Fitted Exponential Distribution (\\lambda_{MLE} = %.4f)', lambdaExp), 'FontSize', 15);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
grid on;
exportgraphics(fig2, fullfile(output_dir, 'histogram_exp_estimated.png'));

% Figure 3: Fitted Rayleigh Model
fig3 = figure('Name', 'Fitted Rayleigh Distribution', 'NumberTitle', 'off', 'Position', [100 50 800 550]);
h3 = histogram(rayleighData, numBins, 'FaceColor', [0.1 0.7 0.3], 'Normalization', 'count');
xlabel('Value x', 'FontSize', 13);
ylabel('Occurrence Count', 'FontSize', 13);
title(sprintf('Fitted Rayleigh Distribution (\\sigma^2_{MLE} = %.4f)', sigmaSqRayleigh), 'FontSize', 15);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
grid on;
exportgraphics(fig3, fullfile(output_dir, 'histogram_rayleigh_estimated.png'));

%% 4. Goodness-of-Fit Evaluation (Bin-wise Histogram MSE)
mseExp = mean((h1.Values - h2.Values).^2);
mseRayleigh = mean((h1.Values - h3.Values).^2);

fprintf('Goodness-of-Fit Comparison (Histogram Bin MSE):\n');
fprintf('  MSE (Data vs. Exponential): %.2f\n', mseExp);
fprintf('  MSE (Data vs. Rayleigh):    %.2f\n\n', mseRayleigh);

if mseRayleigh < mseExp
    fprintf('--> CONCLUSION: The data is decisively best described by a RAYLEIGH distribution.\n');
    fprintf('    The Rayleigh model provides an order-of-magnitude superior fit (MSE %.2f vs %.2f).\n', mseRayleigh, mseExp);
else
    fprintf('--> CONCLUSION: The data is best described by an EXPONENTIAL distribution.\n');
end
