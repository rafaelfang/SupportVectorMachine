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


%% initialization of SVM
[m,k]=size(trainAttribute);
alpha=rand(m,1);
eta=0.01;




%% use gradient descent method
flag=1;
while flag==1
    for i=1:m
        val=eta*kernelFunction(trainAttribute(i),trainAttribute(i));
        if(val>0&&val<2)
            flag=0;
            break;
        end
        sumVal=0;
        for j=1:m
            sumVal=sumVal+alpha(j)*trainLabel(j)*kernelFunction(trainAttribute(i),trainAttribute(j));
        end
        beta=alpha(i)+eta*(1-trainLabel(i)*sumVal);
        alpha(i)=beta*sign(beta);
    end
    


end
%% get overall cost
costAll=0;
for i=1:m
    for j=1:m
        costAll=costAll+alpha(i)*alpha(j)*trainLabel(i)*trainLabel(j)*kernelFunction(trainAttribute(i),trainAttribute(j));
    end
end
H=sum(alpha)-(1/2)*costAll;
optimizedWeight=zeros(1,k);
for i=1:m
   optimizedWeight=optimizedWeight+alpha(i)*trainLabel(i)*trainAttribute(i,:); 
end
