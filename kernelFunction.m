function [ kernelVal ] = kernelFunction( i,j,trainAttribute )
% this is used to calculate the kernel value
% written by Chao Fang
% i, j means the ith and jth sample from training dataset
xi=trainAttribute(i,:);
xj=trainAttribute(j,:);
%linear kernel
kernelVal=xi*xj';

end

