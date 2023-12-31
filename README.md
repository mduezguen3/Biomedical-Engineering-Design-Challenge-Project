# Biomedical-Engineering-Design-Challenge-Project Computational Model

# MATLAB Files

This repository contains MATLAB files for a biomedical engineereing design challenge focused on designing a wearable device for postpartum pregnant women in rural Georgia (U.S.). The device aims to measure vital signs, including heart raste and blood pressure.

##  Interpretation and Exploration

- 'research1.m': MATLAB code analyzes PPG sensor data from wrist and finger, featuring machine learning toolbox in MATLAB functions for data cleaning, raw sensor plots, and Beats Per Minute (BPM) calculation. Linear regression identifies blood pressure trends, and residuals guide model accuracy assessment.

Figure 1: Provides a visual of signal values changing over time, aiding in physiological insights. BPM calculation offers quick heart rate assessment, with potential for refining BPM algorithms for enhanced accuracy.

Figures 2 and 3: Generate linear regression subplots for blood pressure trends. Residual analysis identifies model shortcomings, guiding an iterative refinement process for improved accuracy. Overall, these MATLAB files offer insights into cardiovascular activity, emphasizing the importance of iterative refinement for accurate health monitoring.

## Images 

- MATLAB (Cleaned Raw Wrist Data) Signal Value vs Time (Raw Data), Signal Value vs Time (Blood Pressure Trend), Residual Value vs Time (Residuals).png
  ![MATLAB (Cleaned Raw Wrist Data) Signal Value vs Time (Raw Data), Signal Value vs Time (Blood Pressure Trend), Residual Value vs Time (Residuals)](https://github.com/mduezguen3/Computational-Neuroscience-Research/assets/131891739/0f13fe4f-a640-44d0-b676-f41bc1946234)

- MATLAB (Cleaned Raw Finger Data) Signal Value vs Time (Raw Data), Signal Value vs Time (Blood Pressure Trend), Residual Value vs Time (Residuals).png
  ![MATLAB (Cleaned Raw Finger Data) Signal Value vs Time (Raw Data), Signal Value vs Time (Blood Pressure Trend), Residual Value vs Time (Residuals)](https://github.com/mduezguen3/Computational-Neuroscience-Research/assets/131891739/08ae87df-546b-4acb-ad5a-306e3cc350a7)

- MATLAB (Cleaned BPM Data) Signal Value vs Time (Raw Data), Signal Value vs Time (Blood Pressure Trend), Residual Value vs Time (Residuals).png
  ![MATLAB (Cleaned BPM Data) Signal Value vs Time (Raw Data), Signal Value vs Time (Blood Pressure Trend), Residual Value vs Time (Residuals)](https://github.com/mduezguen3/Computational-Neuroscience-Research/assets/131891739/7d4a314b-acd9-47b3-9c89-8315e350d91a)

##  Interpretation and Exploration

- 'statsapp10.m':  This MATLAB script performs the following statistical analyses on the clean BPM dataset:
Descriptive statistics: Mean, median, mode, variance, standard deviation, standard error, range, correlation coefficient, and linear regression coefficients.
Two-sample t-test to assess the difference between the sample mean and hypothesized population mean (from the Apple Watch).
Calculation of confidence intervals.

## Images 

- MATLAB Finger Raw Data Output.png
  ![MATLAB Finger Raw Data Output](https://github.com/mduezguen3/Computational-Neuroscience-Research/assets/131891739/0ede6534-67fa-45a7-97fa-6b76d04c1465)
- MATLAB Wrist Raw Data Output.png
  ![MATLAB Wrist Raw Data Output](https://github.com/mduezguen3/Computational-Neuroscience-Research/assets/131891739/23498cd8-f39d-4a90-8e0c-3fe461addc6f)
- MATLAB Clean Finger BPM Data Output.png
  ![MATLAB Finger Raw Data Output](https://github.com/mduezguen3/Computational-Neuroscience-Research/assets/131891739/f5f1e1d9-d043-431b-a2a2-444a5f1b5952)
- MATLAB Clean Wrist BPM Data Output.png
  ![MATLAB Wrist BPM Data Output](https://github.com/mduezguen3/Computational-Neuroscience-Research/assets/131891739/9b571b75-63a9-402b-b795-a73e0df66943)

##  Interpretation and Exploration

- 'ttestpval.m': This script executes a t-test to evaluate the significance of the observed sample mean compared to the assumed population mean. It calculates the T-Value, representing the standard errors between the sample mean and population mean, and the P-Value, indicating the probability of observing the given T-Value under the null hypothesis. This script provides insights into the characteristics of datasets, examining the reliability of sample statistics, detecting trends (e.g., blood pressure trends), and assessing the significance of data concerning specific population parameters. It underscores the significance of the t-test as a powerful tool for hypothesis testing and drawing meaningful conclusions from the dataset.

## Images 

- MATLAB Wrist and Finger BPM Data TTest and P-Value Test Output.png
  ![MATLAB Wrist and Finger BPM Data TTest and P-Value Test Output - Copy](https://github.com/mduezguen3/Biomedical-Engineering-Design-Challenge-Project/assets/131891739/85c40d87-1848-4b6c-a5a6-bdd30b196cf4)
  
##  Interpretation and Exploration

- 'aimodel.m':  This MATLAB script performs machine learning toolbox applications in MATLAB by loading PPG data, applying a Butterworth bandpass filter, performing Z-score normalization, and aligning signals using cross-correlation. Afterwards,  defines and trains an LSTM-based autoencoder for feature extraction and ABP waveform reconstruction. Furthermore, the script allows for visualizations of raw and filtered PPG signals, aligned MAP signals, and predicted continuous MAP waveforms to assess the model's performance. Moreover, loads new PPG data, filters and normalizes it, makes predictions using the trained LSTM autoencoder, and visualizes the actual and predicted MAP signals. 

## Images 

- Trained Data - PPG Signal and Filtered PPG and Training Data - Synthetic MAP Signal and Aligned MAP.png
  ![Training Data - PPG Signal and Filtered PPG](https://github.com/mduezguen3/Biomedical-Engineering-Design-Challenge-Project/assets/131891739/b6befb05-b779-4813-b33b-515bad83a599)
- Testing Data - Actual MAP Signals, Testing Data - Predicted MAP Signals, and Testing Data - Difference between Predicted and Actual MAP.png
  ![Testing Data - Actual MAP Signals, Testing Data - Predicted MAP Signals, and Testing Data - Difference between Predicted and Actual MAP](https://github.com/mduezguen3/Biomedical-Engineering-Design-Challenge-Project/assets/131891739/14da17fd-6cec-42e2-bd05-a51dc303cd28)
- Trained AI/Machine Learning Model.png
  ![Training AI model](https://github.com/mduezguen3/Biomedical-Engineering-Design-Challenge-Project/assets/131891739/208e3763-cab8-4f69-942c-af57e4c746c2)






