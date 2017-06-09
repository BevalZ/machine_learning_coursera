function [a,b]=fishsort(x,n,ad)
% This function takes an vector as input and returns the largest or
% smallest n numbers and their indiceds as output
if nargin==1
    n=1;
    ad='accending';
end
if nargin==2
    if isequal(class(n),'char')
        ad=n;
        n=1;
    else
        ad='accending';
    end
end
if n>length(x)
    error('n has to be less than the length of x');
end
a=zeros(n,1);
b=zeros(n,1);
if isequal(ad,'accending')
for i=1:n
    a(i)=min(x);
    [~,b(i)]=min(x);
    x(b(i))=inf;
end
else
    for i=1:n
    a(i)=max(x);
    [~,b(i)]=max(x);
    x(b(i))=-inf;
    end
end
end