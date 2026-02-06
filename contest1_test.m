%% Clean up and import useful functions
clc;clear;close all;
addpath('../functions')

%% Add paths and load data from file 
% We will replace these lines to point to our dataset for testing
% test_set_dummy.txt is a copy of the provided training set, trimmed to
% match the dimensions of the real test set.)
test_set = readmatrix('test_set_dummy.txt'); 

%% Compute cyclostationary mean and std deviation
T  = 365; % period (days)
f = 10; % semi-amplitude of a window size of 21 days

test_set_m = zeros(size(test_set));
test_set_sd = zeros(size(test_set));

for i = 1:size(test_set,2)
    temp = test_set(:,i);
    [~, m] = moving_average(temp, T, f);
    test_set_m(:,i) = m;
    [~, s2] = moving_average( ( temp - m ).^2, T, f);
    s = sqrt(s2);
    test_set_sd(:,i) = s;
end

% compute deseasonalized variables
test_set_deseasonalized = (test_set - test_set_m)./test_set_sd;

%% Separate the training set in inputs and target (x and y)
x = test_set_deseasonalized(:,1:end-1);
y = test_set_deseasonalized(:,end);

%% Forecasting the cumulative 5-day inflow to Hoa Binh
% In this section, you need to load the chosen model and perform 
% forecasting for the output variable on a hypothetical test set, 
% randomly generated from the training set. This ensures that your model 
% can handle a different number of rows.

%_______________________________________________________________
% Nomenclature:
% q_ -> forecast 
% q  -> target variable (Hoa Binh cumulative 5 day inflow)
%_______________________________________________________________

% (1) Load model
load('Deliverable_project_1/output_forecast_train_25.mat','netLSTM');  % Load trained LSTM model

% (2) Compute forecast
% Prepare test sequences for LSTM (sliding window)
seqLen = 20;  % must match training seqLen
XTest = {};
yTest = [];

for i = 1:(length(y) - seqLen)
    Xi = x(i:i+seqLen-1, :)';
    Yi = y(i + seqLen);
    
    if ~any(isnan(Xi), 'all') && ~isnan(Yi)
        XTest{end+1} = Xi;
        yTest(end+1) = Yi;
    end
end

yTest = yTest(:);

%% Predict with LSTM
yPred = predict(netLSTM, XTest)';
yPred = yPred(:);  % ensure column vector

% Reselect correct part of mean and std for deseasonalization
q_ = yPred .* test_set_sd(seqLen+1:end, end) + test_set_m(seqLen+1:end, end);  % predicted cumulative inflow
q  = test_set(seqLen+1:end, end);  % true cumulative inflow

% Plot results
figure
plot([q q_], '.-')
legend('True','Predicted')
title('LSTM Forecast vs Actual')
xlabel('Time Step')
ylabel('5-day Cumulative Inflow')

%% Compute R² score
RSS = sum((q - q_).^2);  % Residual sum of squares
TSS = sum((q - mean(q)).^2);  % Total sum of squares
R2 = 1 - RSS / TSS;

fprintf('R² on test set: %.4f\n', R2);


%% Save forecast and score
% Store the predictions in a column vector and save them as 
% "output_forecast_test_GroupNumber.txt" in the Deliverable_project_1 folder.  
% Save the R² metric in "R2_score_test_GroupNumber.txt" in the same folder.

save('Deliverable_project_1/output_forecast_test_25.mat', 'netLSTM');

fid = fopen('Deliverable_project_1/R2_score_test_25.txt', 'w');
fprintf(fid, '%.4f\n', R2);
fclose(fid);

