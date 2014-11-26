function [ output_args ] = SMO( C,tol,eps,trainAttribute, trainLabel )
%SMO based on pseducode in Platt 1988
%this is the main routine
%implement by Chao Fang
m=size(trainAttribute,1);
numChanged=0;
examineAll=1;
while numChanged > 0 || examineAll==1
    numChanged=0;
    if examineAll==1
        for k=1:m
            numChanged=numChanged+examineExample(k);
        end
    else
        for k=1:m
            if alpha(k)~=0&&alpha(k)~=C
                numChanged=numChanged+examineExample(k);
            end
        end
    end
    if examineAll==1
        examineAll=0;
    elseif numChanged==0
        examineAll=1;
    end
end

end

