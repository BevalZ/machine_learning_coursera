function plot_logistic(max_iteration)
X=load('logistic_x.txt');
Y=load('logistic_y.txt');
figure;
hold on;
X=[ones(size(X,1),1) X]; %Add a column of 1
plot(X(Y>0,2),X(Y>0,3),'go','linewidth',2);% divide X according yo y
plot(X(Y<0,2),X(Y<0,3),'rx','linewidth',2);
X1=min(X(:,2)):0.1:max(X(:,2));
[Theta,J]=logistic_regression(X,Y,max_iteration);
X2=-Theta(2)*X1/Theta(3)-Theta(1)/Theta(3);
plot(X1,X2,'linewidth',2);
xlabel('x1');
ylabel('x2');
legend('positive','negative');
hold off;
figure;
plot(1:max_iteration,J);
fprintf('the theta is:\n');
Theta
end