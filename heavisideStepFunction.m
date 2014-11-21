function [ output ] = heavisideStepFunction( input )
% this is the Heaviside Step Function
% written by Chao Fang

output=zeros(size(input));

for i=1:size(input,2)
    if input(i)>0
        output(i) = 1;
    else 
        output(i) =0;
    end
end
end