
close all
clear
clc
%set random generator seed
rng(1);

%% prepare training and testing dataset
% load diabetes
% 
% trainIndex=1:round(0.8*size(diabetes,1));
% trainAttribute=diabetes(trainIndex,2:end);
% trainLabel=diabetes(trainIndex,1);
% 
% testIndex=round(0.8*size(diabetes,1))+1:size(diabetes,1);
% testAttribute=diabetes(testIndex,2:end);
% testLabel=diabetes(testIndex,1);
% N=size(trainAttribute,1);


trainAttribute=[ 0.5 0.5;
                0.25 0.25;
                1.5 1.5 ;
                1.25 1.25 ];
                
trainAttribute=normc(trainAttribute);
trainLabel=[1;
    1;
    -1;
    -1];
N=4;
%% SMO
C=0.9;
tol=100;
max_passes=100;
[ alpha, b ] = SimplifiedSMO( C, tol, max_passes, trainAttribute, trainLabel );
w=0;
b=zeros(N,1);
for i=1:N
    w=w+alpha(i)*trainLabel(i)*trainAttribute(i,:);
    b(i)=(1/trainLabel(i))-w*trainAttribute(i,:)';
end


bVal=mean(b);

%% training Accuray

predTrain=zeros(N,1);
for i=1:N
    predTrain(i,1)=sign(w*trainAttribute(i,:)'+bVal);
end

trainAcc=sum(trainLabel(:)==predTrain(:))/N;

%% testing Accuracy
testIndex=round(0.8*size(diabetes,1))+1:size(diabetes,1);
testAttribute=diabetes(testIndex,2:end);
testLabel=diabetes(testIndex,1);    
predTest=zeros(size(testAttribute,1),1);
for i=1:size(testAttribute,1)
    predTest(i,1)=sign(w*trainAttribute(i,:)'+bVal);
end  

testAcc=sum(testLabel(:)==predTest(:))/size(testAttribute,1);

