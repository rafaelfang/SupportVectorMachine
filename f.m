function [ fVal ] = f(x,trainAttribute,trainLabel,alpha,b)
%F
%written by Chao Fang
fVal=0;
m=size(trainAttribute,1);
for i=1:m
    fVal=fVal+alpha(i)*trainLabel(i)*kernelFunction( trainAttribute(i,:),x );
end
fVal=fVal+b;
end

