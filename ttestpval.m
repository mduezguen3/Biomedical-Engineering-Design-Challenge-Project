filename = 'wristcleandata1.txt';
data_table = readtable(filename);
data = data_table{:,1};

%threshold decided by group 
popMean = 46;

%we will be performing a one sample ttest
[h, p, ci, stats] = ttest(data, popMean);

fprintf('T-Value: %.4f\n', stats.tstat);
fprintf('P-Value: %.4f\n', p);

if p < 0.05
    fprintf('The result is significant at p < 0.05.\n');
else
    fprintf('The result is not significant at p < 0.05.\n');
end
