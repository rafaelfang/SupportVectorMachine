function [ output ] = heavisideStepFunction( input )
% this is the Heaviside Step Function
% written by Chao Fang

    if input>0
        output=1;
    else
        output=0;
    end
end