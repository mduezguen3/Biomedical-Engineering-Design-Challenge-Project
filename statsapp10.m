function [outputs] = statsapp10(filename)
% Read data from the text file
data_table = readtable(filename);
% Extract the numeric data from the table
data = data_table{:, 1}; % Assuming the data is in the first column
% Descriptive statistics
mean_value = mean(data);
median_value = median(data);
mode_value = mode(data);
variance_value = var(data);
std_deviation_value = std(data);
standard_error_value = std_deviation_value / sqrt(length(data));
range_value = range(data);
% Sample size
sample_size = length(data);
% Correlation and regression
x = (1:length(data))'; % Use row numbers as x
y = data;
% Correlation coefficient
correlation_coefficient = corr(x, y);
% Linear regression
regression_coefficients = polyfit(x, y, 1);
% Confidence interval and p-value
% Confidence interval
confidence_interval = 1.96 * std_deviation_value / sqrt(length(data));
confidence_interval_low = mean_value - confidence_interval;
confidence_interval_high = mean_value + confidence_interval;
[h, p_value] = ttest(data);
% Create strings with values
mean_str = sprintf('Mean: %.6f\n', mean_value);
median_str = sprintf('Median: %.6f\n', median_value);
mode_str = sprintf('Mode: %.6f\n', mode_value);
variance_str = sprintf('Variance: %.6f\n', variance_value);
stddeviation_str = sprintf('Standard Deviation: %.6f\n', std_deviation_value);
standarderror_str = sprintf('Standard Error: %.6f\n', standard_error_value);
range_str = sprintf('Range: %.6f\n', range_value);
correlation_str = sprintf('Correlation Coefficient: %.6f\n',
correlation_coefficient);
regression_str = sprintf('Regression Coefficients: %.6f, %.6f\n',
regression_coefficients(1), regression_coefficients(2));
confidenceinterval_str = sprintf('Confidence Interval: [%.6f,.6f]\n', confidence_interval_low, confidence_interval_high);
pval_str = sprintf('P-Value: %.6f\n', p_value);
samplesize_str = sprintf('Sample Size: %d\n', sample_size);
% Combine the strings
outputs = [mean_str, median_str, mode_str, variance_str,
stddeviation_str, standarderror_str, range_str, correlation_str,
regression_str, confidenceinterval_str, pval_str, samplesize_str];
% Display results
disp(outputs);
end