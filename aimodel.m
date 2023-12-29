% Step 1: Load PPG data for training
ppg_data_train = load('Finger_Data.txt');

% Step 2: Preprocessing - Butterworth bandpass filter
fs = 1; % Adjust the sampling rate based on your data
nyquist = fs / 2;
desired_band = [0.5 8]; % Adjust the desired band based on your data
cutoff_band = desired_band / nyquist;

% Ensure that the cutoff frequencies are within the interval of (0, 1)
cutoff_band = max(min(cutoff_band, 0.99), 0.01);

[b, a] = butter(4, cutoff_band, 'bandpass');
filtered_ppg_train = filtfilt(b, a, ppg_data_train);

% Step 3: Z-score normalization
zscore_ppg_train = zscore(filtered_ppg_train);

% Step 4: Signal alignment using cross-correlation
% Have MAP data for training, load it, and perform cross-correlation
map_data_train = generate_synthetic_map(length(ppg_data_train), fs);
[~, lag_train] = xcorr(zscore_ppg_train, map_data_train);
[max_corr_train, max_index_train] = max(abs(lag_train));

% Align the signals based on the maximum correlation
aligned_map_train = circshift(map_data_train, [0, max_index_train]);

% Save the synthetic MAP data for later use
dlmwrite('synthetic_map_train.txt', aligned_map_train, 'precision', 10);

% Step 5: Model building - LSTM-based autoencoder
% You may need to adjust the architecture and hyperparameters based on your data
input_size = 1;
hidden_size = 50;

% Reshape the input data to have a feature dimension of 1
zscore_ppg_train_reshaped = reshape(zscore_ppg_train, 1, [], 1);

layers = [ ...
    sequenceInputLayer(1) % Change input size to 1
    lstmLayer(hidden_size, 'OutputMode', 'sequence')
    fullyConnectedLayer(1) % Change output size to 1
    regressionLayer];

options = trainingOptions('adam', 'MaxEpochs', 50, 'Plots', 'training-progress');

% Scale and shift aligned ABP data to the desired range
map_data_range = [70, 100];
aligned_map_train_scaled = aligned_map_train - min(aligned_map_train);
aligned_map_train_scaled = aligned_map_train_scaled / max(aligned_map_train_scaled) * (map_data_range(2) - map_data_range(1)) + map_data_range(1);

% Train the LSTM autoencoder
lstm_autoencoder = trainNetwork(zscore_ppg_train_reshaped, aligned_map_train_scaled, layers, options);

% Visualization for training data - PPG (example)
figure;
subplot(2, 1, 1);

% Plot raw PPG data
plot(ppg_data_train, 'b', 'LineWidth', 1);
hold on;

% Plot filtered PPG data
plot(filtered_ppg_train, 'r', 'LineWidth', 0.5);

legend('Raw PPG', 'Filtered PPG');
title('Training Data - PPG Signal and Filtered PPG');

% Visualization for training data - MAP (example)
subplot(2, 1, 2);

% Plot raw MAP data
plot(map_data_train, 'b', 'LineWidth', 1);
hold on;

% Plot aligned MAP data
plot(aligned_map_train, 'r', 'LineWidth', 0.5);

legend('Raw MAP', 'Aligned MAP');
title('Training Data - Synthetic MAP Signal and Aligned MAP');

% Making predictions on new data and visualizing the results
ppg_data_test = load('Wrist_Data.txt');
filtered_ppg_test = filtfilt(b, a, ppg_data_test);
zscore_ppg_test = zscore(filtered_ppg_test);
zscore_ppg_test_reshaped = reshape(zscore_ppg_test, 1, [], 1);

% Load the synthetic MAP data for testing
synthetic_map_test = load('synthetic_map_train.txt');

% Making predictions on new data and visualizing the results
% Replace 'Wrist_Data.txt' and 'actual_abp_data.txt' with your actual data
ppg_data_test = load('Wrist_Data.txt');
filtered_ppg_test = filtfilt(b, a, ppg_data_test);
zscore_ppg_test = zscore(filtered_ppg_test);
zscore_ppg_test_reshaped = reshape(zscore_ppg_test, 1, [], 1);

% Make predictions using the trained LSTM autoencoder
predicted_map_test = predict(lstm_autoencoder, zscore_ppg_test_reshaped);

% Use the synthetic MAP data as the "actual" MAP data
actual_map_data = generate_synthetic_map(length(ppg_data_test), fs); % Adjust as needed

% Scale and shift actual MAP data to the desired range
map_data_range = [70, 90];
actual_map_data_scaled = actual_map_data - min(actual_map_data);
actual_map_data_scaled = actual_map_data_scaled / max(actual_map_data_scaled) * (map_data_range(2) - map_data_range(1)) + map_data_range(1);

% Scale and shift predicted MAP data to the desired range
predicted_map_test_scaled = predicted_map_test - min(predicted_map_test);
predicted_map_test_scaled = predicted_map_test_scaled / max(predicted_map_test_scaled) * (map_data_range(2) - map_data_range(1)) + map_data_range(1);

% Visualize the actual MAP signals for testing data
figure;
subplot(3, 1, 1);
plot(actual_map_data_scaled, 'r', 'LineWidth', 1.5);
legend('Actual MAP');
title('Testing Data - Actual MAP Signals');

% Visualize the predicted MAP signals for testing data
subplot(3, 1, 2);
plot(predicted_map_test_scaled, 'b', 'LineWidth', 1);
legend('Predicted MAP');
title('Testing Data - Predicted MAP Signals');

% Compute the difference between predicted and actual MAP signals
map_difference = predicted_map_test - actual_map_data;

% Visualize the difference between predicted and actual MAP signals for testing data
subplot(3, 1, 3);
plot(map_difference, 'k', 'LineWidth', 1); % Black line for the difference
legend('Difference (Predicted - Actual)');
title('Testing Data - Difference between Predicted and Actual MAP');

function map_data = generate_synthetic_map(num_samples, fs)
    % Generate synthetic ABP data for illustration purposes
    t = (0:num_samples-1) / fs;
    
    % Create a synthetic pulse waveform
    pulse_waveform = sin(2 * pi * 1.5 * t) + 0.5 * sin(2 * pi * 2.5 * t);
    
    % Create a synthetic ABP signal by integrating the pulse waveform
    map_data = cumsum(pulse_waveform);
    
    % Add noise to the synthetic ABP signal
    noise = 0.2 * randn(size(map_data));
    map_data = map_data + noise;
    
    % Scale and shift to achieve the desired MAP range (50 to 80 mmHg)
    map_range = [70, 100];
    map_data = map_data - min(map_data);
    map_data = map_data / max(map_data) * (map_range(2) - map_range(1)) + map_range(1);
end
