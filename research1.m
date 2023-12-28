function [out1] = research1(filename)
   % First let's load raw data from the text file
   fid = fopen(filename, 'r');
   raw_data = textscan(fid, '%f');
   fclose(fid);
   raw_data = raw_data{1}; % Extract the data from the cell array
   % Define the time vector (assuming each data point is sampled at a constant rate)
   time = 1:length(raw_data);
   % Ensure that time and raw_data have the same length
   min_length = min(length(time), length(raw_data));
   time = time(1:min_length);
   raw_data = raw_data(1:min_length);
   % Plot the raw data
   figure;
   subplot(3,1,1);
   plot(time, raw_data);
   title('Raw Data');
   xlabel('Time');
   ylabel('Signal Value');
   % Calculate BPM (Beats Per Minute) using statistical toolbox
   threshold = 550;
   above_threshold = raw_data > threshold;
   bpm = sum(above_threshold) / length(above_threshold) * 60;  % Assuming each data point corresponds to 1 second
   disp(['BPM: ' num2str(bpm)]);
   % Perform linear regression for blood pressure trending/waves
   subplot(3,1,2);
  
   % Fit the model using polyfit
   p = polyfit(time, raw_data, 1);
  
   % Plot the regression line
   plot(time, polyval(p, time), time, raw_data);
   title('Blood Pressure Trend');
   xlabel('Time');
   ylabel('Signal Value');
   % Display the residuals
   residuals = raw_data - polyval(p, time);
   subplot(3,1,3);
   plot(time, residuals);
   title('Residuals');
   xlabel('Time');
   ylabel('Residual Value');
   out1 = filename;
   % Additional machine learning tasks can be performed using the mdl object
end



