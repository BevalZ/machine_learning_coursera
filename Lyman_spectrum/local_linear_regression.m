function yhat = local_linear_regression(x, y, tau)
% LOCAL_LINEAR_REGRESSION Performs a local linear regression to smooth the
% given input signal.
%
% yhat = local_linear_regression(x, y, tau) takes as input the vectors x
% and y, both of the same dimension. Then, at each point x in the given
% vector, fits a local linear regression using the features (1, x) at that
% Figure 5: Resulting functional regression for test set example 6.
% point, with weights given by
%
% w^i(x) = exp(-(x - x^i)^2 / (2 * tau^2)),
%
% that is, transforms the input so that
%
% yhat(i) = [1, x(i)] * theta^(i)
%
% where theta^(i) minimizes
%
% J_i(theta) = sum_{j=1}^m w^j(x(i)) * (y(j) - [1 x(j)] * theta)^2.
if (length(x) ~= length(y))
  error('Length of x (%d) not same as y (%d)¡¯, length(x), length(y)');
end
nn = length(x);
X = [ones(nn, 1), x];
yhat = zeros(nn, 1);
for ii = 1:nn
  w = exp(-(x - x(ii)).^2 / (2 * tau^2));
  XwX = X' * ([w, w] .* X);
  XtWy = X' * (w .* y);
  theta = XwX \ XtWy;
  yhat(ii) = [1 x(ii)] * theta;
end
end