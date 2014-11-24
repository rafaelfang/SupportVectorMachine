function [ alpha, b ] = SMO( C, tol, max_passes, trainAttribute, trainLabel )
%SMO method from Paper "Fast Training of SVM using SMO" by John C. Platt
% this is the main routine for SMO
% written by Chao Fang


m=size(trainAttribute,1);
alpha=zeros(m,1);%lagrangian multiplier
alphaOld=zeros(m,1);
b=0;%threshold for solution
passes=0;
E=zeros(m,1);

while(passes<max_passes)
    num_changed_alphas=0;

    for i=1:m
        E(i)=f(trainAttribute(i,:),trainAttribute,trainLabel,alpha,b)-trainLabel(i);
        if ((trainLabel(i)*E(i)<-tol && alpha(i)<C )||...
                (trainLabel(i)*E(i)>tol && alpha(i)>0))
            j=randi(m);
            while(j==i)
                j=randi(m);
            end
            E(j)=f(trainAttribute(j,:),trainAttribute,trainLabel,alpha,b)-trainLabel(j);
            alphaOld(i)=alpha(i);
            alphaOld(j)=alpha(j);
            
            if trainLabel(i)~=trainLabel(j)
                L=max(0,alpha(j)-alpha(i));
                H=min(C,C+alpha(j)-alpha(i));
            end
            if trainLabel(i)==trainLabel(j)
                L=max(0,alpha(i)+alpha(j)-C);
                H=min(C,alpha(i)+alpha(j));
            end
            
            if L==H
                continue;
            end
            %compute eta
            eta=2*kernelFunction( trainAttribute(i,:),trainAttribute(j,:) )...
                -kernelFunction( trainAttribute(i,:),trainAttribute(i,:) )...
                -kernelFunction( trainAttribute(j,:),trainAttribute(j,:) );
            if(eta>=0)
                continue;
            end
            
            alpha(j)=alpha(j)-trainLabel(j)*(E(i)-E(j))/eta;
            if alpha(j)>H
                alpha(j)=H;
            elseif alpha(j)<=H&&alpha(j)>=L
                alpha(j)=alpha(j);
            else
                alpha(j)=L;
            end
            if abs(alpha(j)-alphaOld(j)<10^-5)
                continue;
            end
            alpha(i)=alpha(i)+trainLabel(i)*trainLabel(j)*(alphaOld(j)-alpha(j));
            b1=b-E(i)-trainLabel(i)*(alpha(i)-alphaOld(i))...
                *kernelFunction( trainAttribute(i,:),trainAttribute(j,:) )...
                -trainLabel(j)*(alpha(j)-alphaOld(j))...
                *kernelFunction( trainAttribute(i,:),trainAttribute(j,:) );
            b2=b-E(j)-trainLabel(i)*(alpha(i)-alphaOld(i))...
                *kernelFunction( trainAttribute(i,:),trainAttribute(i,:) )...
                -trainLabel(j)*(alpha(j)-alphaOld(j))...
                *kernelFunction( trainAttribute(j,:),trainAttribute(j,:) );
            
            if 0<alpha(i)&&alpha(i)<C
                b=b1;
            elseif 0<alpha(j)&&alpha(j)<C
                b=b2;
            else
                b=(b1+b2)/2;
            end
            num_changed=num_changed+1;
        end%end if
    end%end for
    if num_changed_alphas==0
        passes=passes+1;
    else
        passes=0;
    end
    disp(passes)
end%end while

end

