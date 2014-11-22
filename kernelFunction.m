function [ kernelVal ] = kernelFunction( xi,xj )
% this is used to calculate the kernel value
% written by Chao Fang
% i, j means the ith and jth sample from training dataset

%linear kernel
kernelVal=xi*xj';

end

