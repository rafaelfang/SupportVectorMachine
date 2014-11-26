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


%% use SMO method
max_iteration=10;
tolerance=0.01;
C=1;
[ alpha, b ] = SMO( C, tolerance, max_iteration, trainAttribute, trainLabel );




%% use gradient descent method
max_iteration=10;
[ W ] = gradientDescent(  max_iteration, trainAttribute, trainLabel );



%% SVM
C=0;
[alpha, lambda] = SVM(trainAttribute, trainLabel, C);
