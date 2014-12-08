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



trainAttribute=[ 0.5 0.5;
                0.25 0.25;
                1.5 1.5 ;
                1.25 1.25 ];
                
trainAttribute=normc(trainAttribute);
trainLabel=[1;
    1;
    -1;
    -1];
% [ alpha, b ] = SimplifiedSMO( .1, .1, 1000, trainAttribute, trainLabel )
% supportVectors=trainAttribute(find(alpha~=0),:);
% scatter(trainAttribute(:,1),trainAttribute(:,2));
% hold on
% scatter(supportVectors(:,1),supportVectors(:,2),'r');
%% initialization of SVM
[m,k]=size(trainAttribute);
alpha=rand(m,1);
eta=0.75;




%% use gradient descent method
 flag=1;
% while flag==1
for ind=1:100
    for i=1:m
%         val=eta*kernelFunction(trainAttribute(i,:),trainAttribute(i,:));
%         if(val>0&&val<2)
%             flag=0;
%             break;
%         end
        sumVal=0;
        for j=1:m
            sumVal=sumVal+alpha(j)*trainLabel(j)*kernelFunction(trainAttribute(i,:),trainAttribute(j,:));
        end
%         beta=alpha(i)+eta*(1-trainLabel(i)*sumVal);
%         alpha(i)=beta*sign(beta);
%         disp(i)
        tempB=alpha(i)+eta*(1-trainLabel(i)*sumVal);
        if tempB>0
            alpha(i)=tempB;
        else
            alpha(i)=0;
        end
        
        disp((1-trainLabel(i)*sumVal))
    end
    
end   

% end
%% get overall cost
costAll=0;
for i=1:m
    for j=1:m
        costAll=costAll+alpha(i)*alpha(j)*trainLabel(i)*trainLabel(j)*kernelFunction(trainAttribute(i,:),trainAttribute(j,:));
    end
end
H=sum(alpha)-(1/2)*costAll;
optimizedWeight=zeros(1,k);
for i=1:m
   optimizedWeight=optimizedWeight+alpha(i)*trainLabel(i)*trainAttribute(i,:); 
end


trainAttribute*optimizedWeight';
[-5 -5]*optimizedWeight'
scatter(trainAttribute(:,1),trainAttribute(:,2));