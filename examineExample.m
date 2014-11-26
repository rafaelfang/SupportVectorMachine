function [ returnVal ] = examineExample( i2,alpha,C,trainAttribute,trainLabel,b,tol )
%EXAMINEEXAMPLE 
% implemented by Chao Fang

y2=trainLabel(i2);
alph2=alpha(i2);
E2=f(trainAttribute(i2,:),trainAttribute,trainLabel,alpha,b)-trainLabel(i2);
r2=E2*y2;
if ((r2<-tol && alph2<C )||(r2>tol&&alph2>0))
    if (size(find (alpha~=0&alpha~=C),1)>0)
        i1=randi(m);
        while(i2==i1)
            i1=randi(m);
        end
        if takeStep(i1,i2)==1
            returnVal=1;
        end
    end
    while
        
    end
end

end

