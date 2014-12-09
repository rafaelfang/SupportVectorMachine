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
mu1 = [0 0];
mu2 = [100 100];
SIGMA = [1 1.5; 1.5 3];
r1 = mvnrnd(mu1,SIGMA,100);
r2 = mvnrnd(mu2,SIGMA,100);
figure;
scatter(r1(:,1),r1(:,2),'b');
hold on
scatter(r2(:,1),r2(:,2),'g');
trainAttribute=[ r1;r2];
                
trainAttribute=normc(trainAttribute);
trainLabel=[ones(100,1);-ones(100,1)];
    
% [ alpha, b ] = SimplifiedSMO( .1, .1, 1000, trainAttribute, trainLabel )
% supportVectors=trainAttribute(find(alpha~=0),:);
% scatter(trainAttribute(:,1),trainAttribute(:,2));
% hold on
% scatter(supportVectors(:,1),supportVectors(:,2),'r');



%% initialization of SVM
[m,k]=size(trainAttribute);
alpha=rand(m,1);
eta=2;




%% use gradient descent method
 flag=1;
 max_iter=100;
 C=2;
 ourEps=0.01;
counter=0;
while (flag==1)&&(counter<max_iter)
    flag=0;
    for i=1:m

        if (alpha(i)>0&&alpha(i)<C)
            alphaOld=alpha(i);
            
            sumVal=0;
            for j=1:m
                sumVal=sumVal+alpha(j)*trainLabel(j)*kernelFunction(trainAttribute(i,:),trainAttribute(j,:));
            end
  
            tempB=alpha(i)-eta*(1-trainLabel(i)*sumVal);
            if tempB>0
                alphaNew=tempB;
            else
                alphaNew=0;
            end
            
            if(abs(alphaNew-alphaOld)>ourEps)
                alpha(i)=alphaNew;
                flag=1;
            end
            
%         elseif (alpha(i)>=C)
%             alpha(i)=C;
%         else
%             alpha(i)=0;
        end

    end
    counter=counter+1;
end   

% end
%% get overall cost
N=200;
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

%% color the alpha
supportVectorIndex=find(alpha~=0);
scatter(trainAttribute(supportVectorIndex,1),trainAttribute(supportVectorIndex,2),'r');

