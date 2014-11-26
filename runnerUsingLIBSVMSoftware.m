%a test program to classfy diabetes dataset
%the LIBSVM software is from http://www.csie.ntu.edu.tw/~cjlin/libsvm/#matlab
close all
clear
clc
%set random generator seed
rng(1);

%% prepare training and testing dataset
load diabetes

trainIndex=1:round(0.8*size(diabetes,1));
trainAttribute=diabetes(trainIndex,2:end);
trainLabel=diabetes(trainIndex,1);

testIndex=round(0.8*size(diabetes,1))+1:size(diabetes,1);
testAttribute=diabetes(testIndex,2:end);
testLabel=diabetes(testIndex,1);

%%
% Train the model and get the primal variables w, b from the model

% Libsvm options
% -t 0 : linear kernel
% Leave other options as their defaults 
model = svmtrain(trainLabel, trainAttribute, '-t 0');
w = model.SVs' * model.sv_coef;
b = -model.rho;
if (model.Label(1) == -1)
    w = -w; b = -b;
end
[trainPredictedLabel, trainAccuracy, trainDecisionValues] = svmpredict(trainLabel, trainAttribute, model);
%%
[testPredictedLabel, testAccuracy, testDecisionValues] = svmpredict(testLabel, testAttribute, model);