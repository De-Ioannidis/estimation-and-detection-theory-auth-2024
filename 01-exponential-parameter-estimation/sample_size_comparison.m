% Estimation and Detection Theory - AUTh 2024
% Topic 1: Exponential Parameter Estimation
% Part 3: Asymptotic Performance (IMVUE vs MLE across Sample Sizes N)
% Authors: Dimitrios Ioannidis, Anastasios Theocharis Sotiropoulos
%%
clear; close all; clc;

lambda = 2;              % True parameter value (lambda = 2)
experiments = 5000;      % Number of Monte Carlo experiments

sample_sizes = [10, 20, 50, 100];
for N = sample_sizes
    plot_imvue_vs_mle(N, experiments, lambda);
end

%% Helper function to compare IMVUE and MLE
function plot_imvue_vs_mle(N, experiments, lambda)
    mean_dist = 1 / lambda;
    CRLB = lambda^2 / N;
    
    if exist('normrnd', 'file')
        lambda_IMVUE = normrnd(lambda, sqrt(CRLB), [1, experiments]);
    else
        lambda_IMVUE = lambda + sqrt(CRLB) * randn(1, experiments);
    end

    lambda_MLE = zeros(1, experiments);
    has_stats = exist('exprnd', 'file');
    for i = 1:experiments
        if has_stats
            x_simul = exprnd(mean_dist, 1, N);
        else
            x_simul = -mean_dist * log(rand(1, N));
        end
        lambda_MLE(i) = 1 / mean(x_simul);
    end
    
    % Plot dual histograms
    fig = figure('Name', sprintf('IMVUE vs MLE Comparison (N = %d)', N), 'NumberTitle', 'off', 'Position', [100 50 900 500]);
    
    subplot(1, 2, 1);
    histogram(lambda_IMVUE, 'BinWidth', 0.10, 'FaceColor', [0.2 0.4 0.8]);
    xlabel('Estimated \lambda (IMVUE)', 'FontSize', 13);
    ylabel('Frequency', 'FontSize', 13);
    title(sprintf('Ideal MVUE (N = %d)', N), 'FontSize', 14);
    set(gca, 'FontSize', 11, 'LineWidth', 1.2);
    grid on;
    legend(sprintf('Mean: %.3f\nVar:   %.3f', mean(lambda_IMVUE), var(lambda_IMVUE)), 'Location', 'northeast');
    
    subplot(1, 2, 2);
    histogram(lambda_MLE, 'BinWidth', 0.10, 'FaceColor', [0.85 0.33 0.1]);
    xlabel('Estimated \lambda (MLE)', 'FontSize', 13);
    ylabel('Frequency', 'FontSize', 13);
    title(sprintf('MLE Estimator (N = %d)', N), 'FontSize', 14);
    set(gca, 'FontSize', 11, 'LineWidth', 1.2);
    grid on;
    legend(sprintf('Mean: %.3f\nVar:   %.3f', mean(lambda_MLE), var(lambda_MLE)), 'Location', 'northeast');

    % Export figure to assets
    output_dir = fullfile('..', 'assets', '01-exponential-parameter-estimation');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    exportgraphics(fig, fullfile(output_dir, sprintf('histograms_imvue_mle_N_%d.png', N)));
end
