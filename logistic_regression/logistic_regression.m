function [Theta,J]=logistic_regression(X,y,max_iteration)
Theta=zeros(size(X,2),1);
J=ones(max_iteration,1);
m=length(y);
for ii=1:max_iteration
core=y.*(X*Theta);
J(ii)=(1/m)*sum(log(1+exp(-core)));
fprintf('%d th iteration|cost: %f\n',ii,J(ii));
grad=-(1/m)*X'*(y.*(1./(1+exp(core))));
H=(1/m)*X'*diag((1./(1+exp(core))).*(1./(1+exp(-core))))*X;
Theta=Theta-H\grad;
end
end