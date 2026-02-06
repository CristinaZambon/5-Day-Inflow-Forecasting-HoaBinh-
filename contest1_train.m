%% Clean up and import useful functions
clc;clear;close all;
addpath('functions')

%% Import the dataset to Matlab
training_set = readmatrix('training_set_contest_94_05.txt'); 

% column | variable
% 1  = Hoa Binh inflow [m3/s]
% 2  = Flow at TaBu [m3/s]
% 3  = Flow st LaiChau [m3/s]
% 4  = Flow at NamGiang [m3/s]
% 5  = Flow at YenBai [m3/s]
% 6  = Flow at VaQuang [m3/s]
% 7  = Precipitation at MuongTe [m3/s]
% 8  = Precipitation at TamDuong [m3/s]
% 9 =  Precipitation at Da [m3/s] (spatial average over the river basin)
% 10 = Precipitation at BaoLac [m3/s]
% 11 = Precipitation at BacMe [m3/s]
% 12 = Precipitation at HaGiang [m3/s]
% 13 = Cumulative 5-day inflow at Hoa Binh [m3/s]


%% Prepare the input data that you want to use to train the model
% col is an array containing the indexes of the columns of the inputs
% variables you want to use
col=1:13; % the input selection is performed within the model
% col =[2:3,7:8,10:11,13]
% Not good - col = [1,4:5,7,10,13];
% not good - col=[1,6:8,10:11,13];
% not good - col=[1,4:5,7,12:13];
% not good col = [1:2,4:6,12:13];
% col= [2,4:6,8:10,13];

% 4, 10, 2, 6, 5, 3, 9, 8 correlati a output togliendo la prima
% 1, 8, 5, 3,2,6,4, correlati a prima, togliendo output

% Convert to array
data_set = training_set(:,col);

% plot
N = length(data_set);
t = (1:N)'; 
figure
plot (t, data_set)
xlabel ('time[days]')
ylabel ('[m^3/s]') 
title('Dataset plot - annual patterns before deseasonalization');


%% Compute the cyclostationary mean and standard deviation
T  = 365; % period (days)
f = 10; % semi-amplitude of a window size of 21 days
data_set_m = zeros(size(data_set));
data_set_sd = zeros(size(data_set));

for i = 1:size(data_set,2)
    temp = data_set(:,i);
    [~, m] = moving_average(temp, T, f);
    data_set_m(:,i) = m;
    [~, s2] = moving_average( ( temp - m ).^2, T, f);
    s = sqrt(s2);
    data_set_sd(:,i) = s;
end

% compute deseasonalized variables
data_set_deseasonalized = (data_set - data_set_m)./data_set_sd;
% plot the deseasonalized
figure
plot (data_set_deseasonalized)
xlabel ('time t')

tt = repmat( (1:365)' , N/T, 1 );
figure 
plot(tt, data_set_deseasonalized , '-') 
xlabel('time t [days]')
ylabel('Deseasonalized Dataset [m^3/s]')


%% Separate the training set in inputs and target (x and y)
x = data_set_deseasonalized(:,1:end-1);   % exogenous data (inputs) we decided to use  
y = data_set_deseasonalized(:,end); %output (observed cumulative inflow)

% %% Plot of the correlogram between the inputs and the output OPTIONAL
% %% INPUT VARIABLE SELECTION using MRMR Feature Selection for Forecasting

% inputs = x;
% target = y;
% % Use fsrmrmr to rank the variables
% [idx, scores] = fsrmrmr(inputs, target);
% % Sort the scores in descending order and reorder the indices
% [sorted_scores, sorted_idx] = sort(scores, 'descend');
% % Plot the results
% figure;
% bar(sorted_scores);
% set(gca, 'XTickLabel', sorted_idx, 'XTick', 1:length(sorted_idx));
% xlabel('Variable Number (Sorted by MRMR Rank)');
% ylabel('MRMR Score');
% title('Feature Ranking Based on MRMR - after deseasonalisation');
% grid on;

% % Plot of the correlogram between the inputs and the output 
% idx_flow=[1:6]; %indexes of flow  
% idx_prec=[7:12];  %indexes of precipitation
% 
%       %  corr stream flows-output 
%        for i = 1:length(idx_flow)
%            figure
%             correlogram(x(:,idx_flow(i)),y,30);
%            title('Correlogram flow - cumulative flow')
%         end
% 
%        % corr precipitation-output 
%         for i = 1:length(idx_prec)
%            figure
%            correlogram(x(:,idx_prec(i)),y,50);
%            title('Correlogram precipitation - cumulative flow') 
%         end


% Compute the Pearson correlation matrix
correlation_matrix = corrcoef(data_set_deseasonalized);

figure;
imagesc(correlation_matrix);
colorbar;
title('Pearson Correlation Matrix of Input Variables');
xlabel('Variable Index');
ylabel('Variable Index');
set(gca, 'XTick', 1:12, 'YTick', 1:12);
axis square;

%% Training set and validation set preparation
% set how many years you want to use for training the model and how many
% years you want to use for validation
%%% 30% = 1314 ;;; 70% = 3066

totalYears = (length(data_set) / 365) - 1;
n_y_training = 9; % 70% di training 
n_y_validation = ((totalYears + 1) - n_y_training) ;

% T =365
x_train = x(1:T*n_y_training,:); %exogenous data used for training
x_val = x(T*n_y_training+1:end,:); %exogenous data used for validation
y_train = y(1:T*n_y_training); % cumulative flow  ahead for training 
y_val = y(T*n_y_training+1:end); % cumulative flow  ahead for validation

% Save statistics on cyclostationary
 m = data_set_m(1:T*n_y_training,end); %data of the cyclostationary mean used for training
 s = data_set_sd(1:T*n_y_training,end); 
 mv = data_set_m(T*n_y_training+1:end,end); %data of the cyclostationary mean used for validation
 sv = data_set_sd(T*n_y_training+1:end,end); 

%  %% Input matrix framing
% %lag 2
% for i= 1:(length(data_set_deseasonalized)-1)
% M2 (i, :) = [data_set_deseasonalized(i,1:end-1), data_set_deseasonalized(i+1,:)];
% end
% M2_train = M2(1:T*n_y_training-1,:);
% M2_val = M2(T*n_y_training:end,:);
% 
% %lag 3
% for i=1:(length (data_set_deseasonalized)-2)
% M3 (i, :) = [data_set_deseasonalized(i,1: end-1), data_set_deseasonalized(i+1,1:end-1), data_set_deseasonalized(i+2,:)]; 
% end
% M3_train = M3(1:T*n_y_training-2,:);
% M3_val = M3(T*n_y_training-1:end,:);
% X3 = M3_train(:,1:end-1); % input
% Y3 = M3_train(:,end); % output
% Xv3 = M3_val(:,1:end-1);
% 
% %lag 4
% for i=1:(length(data_set_deseasonalized)-3)
% M4 (i, :) = [data_set_deseasonalized(i,1:end-1), data_set_deseasonalized(i+1,1:end-1), data_set_deseasonalized(i+2,1:end-1), data_set_deseasonalized(i+3,:)];
% end
% M4_train = M4(1:T*n_y_training-3,:);
% M4_val = M4(T*n_y_training-2:end,:);
% 
% %% Train the Linear model
% %LAG2
% 
%  %calibration  
%  out2 = M2_train(:,end);
%  theta2 = M2_train(:,1:end-1)\out2;
%  x_2 = M2_train(:,1:end-1)*theta2;
%  n_2=x_2.*s(2:end, end) + m(2:end, end);
%  R2_c2 = 1 - sum((data_set(2:T*n_y_training,end) - n_2).^2)/sum((data_set(2:T*n_y_training,end) - m(2:T*n_y_training)).^2);
% 
%  %validation
%  output_v2= M2_val(:,1:end-1)*theta2;    
%  n_v2=output_v2.*sv(:,end)+mv(:,end);
%  R2_v2 = 1 - sum((data_set(T*n_y_training+1:end,end) - n_v2).^2)/sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% 
%  R2_lin2=[R2_c2;R2_v2];
% 
%  %LAG3
% 
%  %calibration  
%  out3 = M3_train(:,end);
%  theta3 = M3_train(:,1:end-1)\out3;
%  x_3 = M3_train(:,1:end-1)*theta3;
%  n_3=x_3.*s(3:end, end) + m(3:end, end);
%  R2_c3 = 1 - sum((data_set(3:T*n_y_training,end) - n_3).^2)/sum((data_set(3:T*n_y_training,end) - m(3:T*n_y_training)).^2);
% 
%  %validation
%  output_v3= M3_val(:,1:end-1)*theta3;    
%  n_v3=output_v3.*sv(:,end)+mv(:,end);
%  R2_v3 = 1 - sum((data_set(T*n_y_training+1:end,end) - n_v3).^2)/sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% 
%  R2_lin3=[R2_c3;R2_v3]; 
% 
%  %LAG4
% 
%  %calibration  
%  out4 = M4_train(:,end);
%  theta4 = M4_train(:,1:end-1)\out4;
%  x_4 = M4_train(:,1:end-1)*theta4;
%  n_4=x_4.*s(4:end, end) + m(4:end, end);
%  R2_c4 = 1 - sum((data_set(4:T*n_y_training,end) - n_4).^2)/sum((data_set(4:T*n_y_training,end) - m(4:T*n_y_training)).^2);
% 
%  %validation
%  output_v4= M4_val(:,1:end-1)*theta4;    
%  n_v4=output_v4.*sv(:,end)+mv(:,end);
%  R2_v4 = 1 - sum((data_set(T*n_y_training+1:end,end) - n_v4).^2)/sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% 
%  R2_lin4=[R2_c4;R2_v4];
% 
% 
% %% Train Non-Linear model 
% 
% %% ANN
%  %LAG3 - 3 gg consecutive
% X_ann3= M3_train(:,1:end-1); %exogenous data
% Y_ann3= M3_train(:,end); 
% Xv_ann3= M3_val(:,1:end-1); %ex data for validation
% 
% Nruns=50; % initilaziation of the number of runs the ANN calibration
% R2_ann3=zeros(Nruns,3); %first column training, second column for validation, third column index
% 
% %all_net = cell (Nruns,1);
%  for i=1:Nruns            
% 
%  net_i3=feedforwardnet(4); % not better with 'trainlm'
%  net_i3.trainParam.showWindow = false;
%  net_i3 = train(net_i3,X_ann3',Y_ann3'); %training the networks
% 
%  Yann3_i=[net_i3(X_ann3')']; %This runs the trained ANN on the training inputs to predict outputs.
%  output3_i=Yann3_i.*s(3:end,end)+m(3:end,end); %This reverses the deseasonalization of the ANN outputs
% 
%  R2_ann3(i,1) = 1 - sum((data_set(3:T*n_y_training,end) - output3_i).^2)/sum((data_set(3:T*n_y_training,end) - m(3:T*n_y_training)).^2);
% 
% % all_net{i} = net_i3; %save all networks
% 
%  %validation 
%  Y3i_v=[ net_i3(Xv_ann3')'];
%  q3_iv=Y3i_v.*sv(:,end)+mv(:,end);
% 
%  R2_ann3(i, 2) = 1 - sum((data_set(T*n_y_training+1:end,end) - q3_iv).^2)/sum((data_set(T*n_y_training+1:end,end) - mv).^2);
%  R2_ann3(i,3)=i; %index of the following run
%  end
% 
% %% CART for Lag 3
% % Parameters
% minleafsize_cart = 15;
% alpha = 0.01;
% %LAG 3
% T0_3=fitrtree(X3,Y3);  
% view(T0_3,'mode','graph');
% 
% % (A) EARLY STOPPING
% % training
% T1_3=fitrtree(X3,Y3,'MinLeafSize',minleafsize_cart); 
% view(T1_3,'mode','graph');          
% Y_cart3=[predict(T1_3,X3)];
% n_cart3=Y_cart3.*s(3:end, end)+m(3:end, end);
% R2_cart_c3 = 1 - sum((data_set(3:T*n_y_training,end) - n_cart3).^2)/sum((data_set(3:T*n_y_training,end) - m(3:T*n_y_training)).^2);
% 
% % validation
% Y_cart_v3=[predict(T1_3,Xv3)];
% n_cart_v3=Y_cart_v3.*sv(:,end)+mv(:,end);
% R2_cart_v3= 1 - sum((data_set(T*n_y_training+1:end,end) - n_cart_v3).^2)/sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% 
% R2_cart3=[R2_cart_c3;R2_cart_v3];
% 
% % Automatic routine - OPTIMIZATION OF MinLeafSize
% Topt3= fitrtree(X3, Y3, 'OptimizeHyperparameters', 'auto');                            
% 
% % (B) PRUNING 
% Tpruned3=prune(T0_3,'Alpha',alpha);
% view(Tpruned3, 'mode', 'graph');
% 
% % training
% Y_cart_prun3=[predict(Tpruned3,X3)];
% n_cart_prun3=Y_cart_prun3.*s(3:end,end)+m(3:end, end);
% R2_cart_prun_c3 = 1 - sum((data_set(3:T*n_y_training,end) - n_cart_prun3).^2)/sum((data_set(3:T*n_y_training,end) - m(3:T*n_y_training)).^2);
% 
% % validation
% Y_cart_v_prun3=[predict(Tpruned3,Xv3)];
% n_cart_v_prun3=Y_cart_v_prun3.*sv(:,end)+mv(:,end);
% R2_cart_prun_v3 = 1 - sum((data_set(T*n_y_training+1:end,end) - n_cart_v_prun3).^2)/sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% 
% R2_cart_prun_3=[R2_cart_prun_c3;R2_cart_prun_v3];
% 
% 
% %% RANDOM FOREST
% n_trees_vec = [1 5 50 100 500 1000];  % vector with number of trees
% minleafsize_RF = 20; % minleafsize
% numvar = 10;  % numvariables to sample
% minLeafVals = [5, 10, 20];
% numVarVals = [5, 10, 20];
% R2_RF3_grid = zeros(length(minLeafVals)*length(numVarVals), 3); 
% idx = 1;
% 
% %LAG 3
% R2_RF3 = zeros(length(n_trees_vec), 2);    
% for i = 1:length(n_trees_vec)
%       for ml = minLeafVals
%     for nv = numVarVals
%         tree = templateTree('MinLeafSize', ml, 'NumVariablesToSample', nv);
%         RF = fitrensemble(X3, Y3, 'Method', 'Bag', ...
%             'Learners', tree, 'NumLearningCycles', 500);
% 
%         Y_train = predict(RF, X3);
%         Y_val = predict(RF, Xv3);
% 
%         n_train = Y_train .* s(3:end,end) + m(3:end,end);
%         n_val = Y_val .* sv(:,end) + mv(:,end);
% 
%         R2_RF3_grid(idx, 1) = 1 - sum((data_set(3:T*n_y_training,end) - n_train).^2) / ...
%                                    sum((data_set(3:T*n_y_training,end) - m(3:T*n_y_training)).^2);
%         R2_RF3_grid(idx, 2) = 1 - sum((data_set(T*n_y_training+1:end,end) - n_val).^2) / ...
%                                    sum((data_set(T*n_y_training+1:end,end) - mv).^2);
%         R2_RF3_grid(idx, 3) = idx;
%         idx = idx + 1;
%     end
%       end
% end
% 
% %% testing combo of minleaf and numVar
% minLeafVals = [5, 10, 20];
% numVarVals = [5, 10, 20];
% R2_RF3_grid = zeros(length(minLeafVals)*length(numVarVals), 3); 
% idx = 1;
% 
% 
% 
% 
% %% ANN with lag 2
% % lower R2
% X_ann2 = M2_train(:,1:end-1); % inputs for ANN
% Y_ann2 = M2_train(:,end);     % output for ANN (last column)
% Xv_ann2 = M2_val(:,1:end-1);
% Nruns = 50;
% R2_ann2 = zeros(Nruns,3);
% 
% for i = 1:Nruns
%     net_i2 = feedforwardnet([5 5 5]);
% 
%     net_i2.trainParam.showWindow = false;
%     net_i2 = train(net_i2, X_ann2', Y_ann2');
% 
%     Yann2_i = net_i2(X_ann2')';
%     % Deseasonalize outputs: note indexing from 2 (lag 2)
%     output2_i = Yann2_i .* s(2:end,end) + m(2:end,end);
% 
%     R2_ann2(i,1) = 1 - sum((data_set(2:T*n_y_training,end) - output2_i).^2) / ...
%                         sum((data_set(2:T*n_y_training,end) - m(2:T*n_y_training,end)).^2);
% 
%     % Validation
%     Y2i_v = net_i2(Xv_ann2')';
%     q2_iv = Y2i_v .* sv(:,end) + mv(:,end);
% 
%     R2_ann2(i, 2) = 1 - sum((data_set(T*n_y_training+1:end,end) - q2_iv).^2) / ...
%                         sum((data_set(T*n_y_training+1:end,end) - mv).^2);
%     R2_ann2(i,3) = i;
% 
% end
% 
% 
% %% CART with lag 2
% X2 = M2_train(:,1:end-1);
% Y2 = M2_train(:,end);
% 
% T0_2 = fitrtree(X2, Y2);
% view(T0_2, 'mode', 'graph');
% 
% minleafsize_cart = 15;
% alpha = 0.01;
% 
% T1_2 = fitrtree(X2, Y2, 'MinLeafSize', minleafsize_cart);
% view(T1_2, 'mode', 'graph');
% 
% Y_cart2 = predict(T1_2, X2);
% n_cart2 = Y_cart2 .* s(2:end,end) + m(2:end,end);
% 
% R2_cart_c2 = 1 - sum((data_set(2:T*n_y_training,end) - n_cart2).^2) / ...
%                   sum((data_set(2:T*n_y_training,end) - m(2:T*n_y_training,end)).^2);
% 
% Xv2 = M2_val(:,1:end-1);
% Y_cart_v2 = predict(T1_2, Xv2);
% n_cart_v2 = Y_cart_v2 .* sv(:,end) + mv(:,end);
% 
% R2_cart_v2 = 1 - sum((data_set(T*n_y_training+1:end,end) - n_cart_v2).^2) / ...
%                   sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% 
% R2_cart2 = [R2_cart_c2; R2_cart_v2];
% 
% % Optimization
% Topt2 = fitrtree(X2, Y2, 'OptimizeHyperparameters', 'auto');
% 
% % Pruning
% Tpruned2 = prune(T0_2, 'Alpha', alpha);
% view(Tpruned2, 'mode', 'graph');
% 
% Y_cart_prun2 = predict(Tpruned2, X2);
% n_cart_prun2 = Y_cart_prun2 .* s(2:end,end) + m(2:end,end);
% 
% R2_cart_prun_c2 = 1 - sum((data_set(2:T*n_y_training,end) - n_cart_prun2).^2) / ...
%                        sum((data_set(2:T*n_y_training,end) - m(2:T*n_y_training,end)).^2);
% 
% Y_cart_v_prun2 = predict(Tpruned2, Xv2);
% n_cart_v_prun2 = Y_cart_v_prun2 .* sv(:,end) + mv(:,end);
% 
% R2_cart_prun_v2 = 1 - sum((data_set(T*n_y_training+1:end,end) - n_cart_v_prun2).^2) / ...
%                        sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% 
% R2_cart_prun_2 = [R2_cart_prun_c2; R2_cart_prun_v2];
% 
% 
% %% RANDOM FOREST with lag 2
% n_trees_vec = [1 5 50 100 500 1000];
% minleafsize_RF = 15;
% numvar = 10;
% 
% R2_RF2 = zeros(length(n_trees_vec), 2);
% for i = 1:length(n_trees_vec)
%     tree = templateTree('MinLeafSize', minleafsize_RF, 'NumVariablesToSample', numvar);
%     RF2 = fitrensemble(X2, Y2, 'Method', 'Bag', 'Learners', tree, 'NumLearningCycles', n_trees_vec(i));
% 
%     Y_RF2 = predict(RF2, X2);
%     Y_RF_v2 = predict(RF2, Xv2);
% 
%     n_RF_c2 = Y_RF2 .* s(2:end,end) + m(2:end,end);
%     n_RF_v2 = Y_RF_v2 .* sv(:,end) + mv(:,end);
% 
%     R2_RF2(i,1) = 1 - sum((data_set(2:T*n_y_training,end) - n_RF_c2).^2) / ...
%                        sum((data_set(2:T*n_y_training,end) - m(2:T*n_y_training,end)).^2);
%     R2_RF2(i,2) = 1 - sum((data_set(T*n_y_training+1:end,end) - n_RF_v2).^2) / ...
%                        sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% end
% 
% 
% %% ANN - LAG 4
% X_ann4 = M4_train(:,1:end-1); % exogenous inputs
% Y_ann4 = M4_train(:,end); 
% Xv_ann4 = M4_val(:,1:end-1);  % validation inputs
% 
% Nruns = 50;
% R2_ann4 = zeros(Nruns, 3); % [train, validation, index]
% 
% for i = 1:Nruns
%     net_i4 = feedforwardnet([5 5 5]);
%     net_i4.trainParam.showWindow = false;
%     net_i4 = train(net_i4, X_ann4', Y_ann4');
% 
%     % Training prediction
%     Yann4_i = net_i4(X_ann4')';
%     output4_i = Yann4_i .* s(4:end,end) + m(4:end,end);
%     R2_ann4(i,1) = 1 - sum((data_set(4:T*n_y_training,end) - output4_i).^2) / sum((data_set(4:T*n_y_training,end) - m(4:T*n_y_training)).^2);
% 
%     % Validation
%     Y4i_v = net_i4(Xv_ann4')';
%     q4_iv = Y4i_v .* sv(:,end) + mv(:,end);
%     R2_ann4(i,2) = 1 - sum((data_set(T*n_y_training+1:end,end) - q4_iv).^2) / sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% 
%     R2_ann4(i,3) = i;
% 
%     if R2_ann4(i,2) >= max(R2_ann4(:,2))
%         net_opt4 = net_i4;
%     end
% end
% 
% 
% %% CART - LAG 4
% X4 = M4_train(:,1:end-1);
% Y4 = M4_train(:,end);
% T0_4 = fitrtree(X4,Y4);  
% view(T0_4,'mode','graph');
% 
% % (A) EARLY STOPPING
% T1_4 = fitrtree(X4,Y4,'MinLeafSize',minleafsize_cart); 
% view(T1_4,'mode','graph');          
% Y_cart4 = predict(T1_4,X4);
% n_cart4 = Y_cart4 .* s(4:end,end) + m(4:end,end);
% R2_cart_c4 = 1 - sum((data_set(4:T*n_y_training,end) - n_cart4).^2) / sum((data_set(4:T*n_y_training,end) - m(4:T*n_y_training)).^2);
% 
% Xv4 = M4_val(:,1:end-1);
% Y_cart_v4 = predict(T1_4,Xv4);
% n_cart_v4 = Y_cart_v4 .* sv(:,end) + mv(:,end);
% R2_cart_v4 = 1 - sum((data_set(T*n_y_training+1:end,end) - n_cart_v4).^2) / sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% R2_cart4 = [R2_cart_c4; R2_cart_v4];
% 
% % (B) OPTIMIZATION
% Topt4 = fitrtree(X4, Y4, 'OptimizeHyperparameters', 'auto');
% 
% % (C) PRUNING
% Tpruned4 = prune(T0_4, 'Alpha', alpha);
% view(Tpruned4, 'mode', 'graph');
% 
% Y_cart_prun4 = predict(Tpruned4, X4);
% n_cart_prun4 = Y_cart_prun4 .* s(4:end,end) + m(4:end,end);
% R2_cart_prun_c4 = 1 - sum((data_set(4:T*n_y_training,end) - n_cart_prun4).^2) / sum((data_set(4:T*n_y_training,end) - m(4:T*n_y_training)).^2);
% 
% Y_cart_v_prun4 = predict(Tpruned4, Xv4);
% n_cart_v_prun4 = Y_cart_v_prun4 .* sv(:,end) + mv(:,end);
% R2_cart_prun_v4 = 1 - sum((data_set(T*n_y_training+1:end,end) - n_cart_v_prun4).^2) / sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% 
% R2_cart_prun_4 = [R2_cart_prun_c4; R2_cart_prun_v4];
% 
% %% RANDOM FOREST - LAG 4
% R2_RF4 = zeros(length(n_trees_vec), 2);    
% for i = 1:length(n_trees_vec)
%     tree = templateTree('MinLeafSize',minleafsize_RF, 'NumVariablesToSample', numvar); 
%     RF4 = fitrensemble(X4, Y4, 'Method', 'Bag', 'Learners', tree, 'NumLearningCycles', n_trees_vec(i));
% 
%     Y_RF4 = predict(RF4, X4);
%     Y_RF_v4 = predict(RF4, Xv4);
% 
%     n_RF_c4 = Y_RF4 .* s(4:end,end) + m(4:end,end);
%     n_RF_v4 = Y_RF_v4 .* sv(:,end) + mv(:,end);
% 
%     R2_RF4(i, 1) = 1 - sum((data_set(4:T*n_y_training,end) - n_RF_c4).^2) / sum((data_set(4:T*n_y_training,end) - m(4:T*n_y_training)).^2);
%     R2_RF4(i, 2) = 1 - sum((data_set(T*n_y_training+1:end,end) - n_RF_v4).^2) / sum((data_set(T*n_y_training+1:end,end) - mv).^2);
% end
% 
% 
% 
% % LAG 2
% figure
% bar([R2_c2, R2_v2; R2_cart_c2, R2_cart_v2; R2_ann2(:,1), R2_ann2(:,2); R2_RF2(:,1), R2_RF2(:,2)]);
% xticks(1:4)
% xticklabels({'Linear', 'CART', 'ANN', 'RF'})
% set(gca, 'FontName', 'Imprint MI Shadow')
% legend('Calibration', 'Validation', 'FontName', 'Imprint MT Shadow')
% ylabel('R^2', 'FontName', 'Imprint MT Shadow')
% title('R^2 for calibration and validation for different models (Lag 2)')
% xtickangle(45)
% 
% % LAG 3
% figure
% bar([R2_c3, R2_v3; R2_cart_c3, R2_cart_v3; R2_ann3(:,1), R2_ann3(:,2); R2_RF3(:,1), R2_RF3(:,2)]);
% xticks(1:4)
% xticklabels({'Linear', 'CART', 'ANN', 'RF'})
% set(gca, 'FontName', 'Imprint MI Shadow')
% legend('Calibration', 'Validation', 'FontName', 'Imprint MT Shadow')
% ylabel('R^2', 'FontName', 'Imprint MT Shadow')
% title('R^2 for calibration and validation for different models (Lag 3)')
% xtickangle(45)
% 
% % LAG 4
% figure
% bar([R2_c4, R2_v4; R2_cart_c4, R2_cart_v4; R2_ann4(:,1), R2_ann4(:,2); R2_RF4(:,1), R2_RF4(:,2)]);
% xticks(1:4)
% xticklabels({'Linear', 'CART', 'ANN', 'RF'})
% set(gca, 'FontName', 'Imprint MI Shadow')
% legend('Calibration', 'Validation', 'FontName', 'Imprint MT Shadow')
% ylabel('R^2', 'FontName', 'Imprint MT Shadow')
% title('R^2 for calibration and validation for different models (Lag 4)')
% xtickangle(45)

%% LSTM Implementation
% from 5days till 18 days Sliding Window, R2 increases till 0.84
% over 22 days (25,30), R2 decreases at 0.83
seqLen = 20;
numFeatures = size(x_train, 2);
XTrain = {}; % vector of matrixes to host the data after sliding window
YTrain = [];

% sliding window of 20 days
% LSTM requires sequences with fixed length and a single output
for i = 1:(length(y_train) - seqLen)

    % create a sample Xi as a [13 × 20] matrix (features × timesteps).
    Xi = x_train(i:i+seqLen-1, :)';
   %this is the target value, so the cumulative inflow on the day after the seqLen input days.
    Yi = y_train(i + seqLen);

    if ~any(isnan(Xi), 'all') && ~isnan(Yi)
        XTrain{end+1} = Xi;
        YTrain(end+1) = Yi;
    end
end

XVal = {};
YVal = [];

for i = 1:(length(y_val) - seqLen)
    Xi = x_val(i:i+seqLen-1, :)';
    Yi = y_val(i + seqLen);

    if ~any(isnan(Xi), 'all') && ~isnan(Yi)
        XVal{end+1} = Xi;
        YVal(end+1) = Yi;
    end
end

YTrain = YTrain(:);  % convert to [N × 1]
YVal = YVal(:);      % same for validation

%% Define LSTM architecture
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(40, 'OutputMode','last')  % 40 LSTM units and LASt because we want to predict the inflow after the sequence
    fullyConnectedLayer(1) %maps the LSTM output to a single scalar prediction
    regressionLayer]; % this is standard for continuous value prediction


%% Set training options
% Levenberg-Marquardt (LM) --> a second order method
% Adam optimizer -->  algorithm for first-order gradient-based optimisation that uses adaptive momentum
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ... % time to learn patterns without overfitting
    'L2Regularization', 0.04, ... % EMBEDDED feature selection  / the lower, + overfitting / best is 0.04
    'MiniBatchSize', 64, ...
    'GradientThreshold', 1, ...
    'Shuffle','every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'Plots','none', ...
    'Verbose',false);

%% Train LSTM
% it trains an LSTM to map sequences of your deseasonalized input variables to the deseasonalized cumulative inflow
netLSTM = trainNetwork(XTrain, YTrain, layers, options);

%% Predict and evaluate

% Predict on training data
YPredTrain = predict(netLSTM, XTrain)';
YPredTrain = YPredTrain(:); 

% Revert deseasonalization
YPredTrain_deseason = YPredTrain .* s(seqLen+1:T*n_y_training, end) + m(seqLen+1:T*n_y_training, end);

% True values for training
YTrain_true = data_set(seqLen+1:T*n_y_training, end);

% Calculate R² for training set
RSS_train = sum((YTrain_true - YPredTrain_deseason).^2);  % Residual sum of squares
TSS_train = sum((YTrain_true - mean(YTrain_true)).^2);    % Total sum of squares
R2_train = 1 - (RSS_train / TSS_train);  % R² for training

% Predict on validation data
YPredVal = predict(netLSTM, XVal)';
YPredVal = YPredVal(:);    

% Revert deseasonalization
YPredVal_deseason = YPredVal .* sv(seqLen+1:end, end) + mv(seqLen+1:end, end);

% True values for validation
YVal_true = data_set(T*n_y_training+seqLen+1:end, end);

% Calculate R² for validation set
RSS_val = sum((YVal_true - YPredVal_deseason).^2);  % Residual sum of squares
TSS_val = sum((YVal_true - mean(YVal_true)).^2);    % Total sum of squares
R2_val = 1 - (RSS_val / TSS_val);  % R² for validation

fprintf('R² Training: %.4f\n', R2_train);
fprintf('R² Validation: %.4f\n', R2_val);


%% Save your preferred model to file, along with forecast obtained during training and R2 
% Save model to a file in folder Deliverable_project_1
% Store the predictions in a column vector and save them as 
% "output_forecast_train_GroupNumber.txt" in the Deliverable_project_1 folder.  
% Save the R² metric in "R2_score_train_GroupNumber.txt" in the same folder.


save('Deliverable_project_1/output_forecast_train_25.mat', 'netLSTM');

fid = fopen('Deliverable_project_1/R2_score_train_25.txt', 'w');
fprintf(fid, '%.4f\n', R2_train);
fclose(fid);

