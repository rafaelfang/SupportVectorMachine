close all
clear
clc
%set random generator seed
rng(1);

load diabetes

trainIndex=1:round(0.8*size(diabetes,1));
trainAttribute=diabetes(trainIndex,2:end);
trainLabel=diabetes(trainIndex,1);
% trainAttribute=[ 0.5 0.5;
%                 0.25 0.25;
%                 1.5 1.5 ;
%                 1.25 1.25 ];
%                 
% trainAttribute=normc(trainAttribute);
% trainLabel=[1;
%     1;
%     -1;
%     -1];
% N=4;


N=size(trainAttribute,1);
H=zeros(N,N);
for i=1:N
   for j=1:N
       H(i,j)=trainLabel(i)*trainLabel(j)*kernelFunction( trainAttribute(i,:),trainAttribute(j,:));
       
   end
    
end
lb = zeros(N,1); 
ub = 0.9*ones(N,1);
A=trainLabel';
b=0;
f=-1*ones(N,1);
options = optimoptions('quadprog','Algorithm','interior-point-convex');
[alpha,fval,exitflag,output] = ... 
        quadprog(H,f,A,b,[],[],lb,ub,[],options);

for i=1:size(alpha,1)    
   if(alpha(i)<10^-3)
       alpha(i)=0;
   end
end


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