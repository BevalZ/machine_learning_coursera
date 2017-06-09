function yhat=lwr(x,y,tau,x1)
% This function computed the locally weighted linear regression
%curve for y-x,depending on different tau
if nargin==3
    x1=x;
end
if (size(x,1)>1&&size(x,2)>1)||(size(y,1)>1&&size(y,2)>1)
    error('x and y have to be vectors');
end
if length(x)~=length(y)
    error('x and y have to be of the same length');
end
if min(x1)<min(x)||max(x1)>max(x)
    warning('x1 has better to be within the range of x');
end
if size(x,1)==1
    x=x'; % If x is a row vector, turn it into column vector
end
if size(y,1)==1
    y=y';
end
n=length(x);
yhat=zeros(length(x1),1);
X=[ones(n,1) x];
for i=1:length(x1)
    w=exp(-((x-x1(i)).^2)/(2*tau^2));
    Theta=pinv(X'*([w w].*X))*(X'*(w.*y));
    yhat(i)=Theta(1)+Theta(2)*x1(i);
end
end