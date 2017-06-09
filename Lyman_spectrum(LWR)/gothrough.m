%% Part 1: Initialize
clear ; close all; clc
%% Part 2: Getting the data
urlwrite('http://cs229.stanford.edu/ps/ps1/quasar_train.csv','trainingset.csv');
urlwrite('http://cs229.stanford.edu/ps/ps1/quasar_test.csv','testset.csv');
fprintf('Please hit enter to continue\n');
pause;
%% Part 3: Load the data
%The first row of the matrix is the wavelength, the remains are the data
load('trainingset.csv');
lambda=trainingset(1,:);
training=trainingset(2:end,:);
load('testset.csv');
test=testset(2:end,:);
fprintf('Please hit enter to continue\n');
pause;
%% Part 4: Fit the first training example using unweighted normal equation
firstexample=training(1,:);
figure;
hold on;
plot(lambda,firstexample,'rx','linewidth',2);
xlabel('lambda');
ylabel('intensity');
lambda1=[ones(1,size(lambda,2));lambda];
Theta1=pinv(lambda1*lambda1')*(lambda1*firstexample');
lambdaplot=(min(lambda)-50):1:max(lambda+1);
intensityplot=Theta1(1)+Theta1(2)*lambdaplot;
plot(lambdaplot,intensityplot,'linewidth',2);
legend('rawdata','line');
hold off;
fprintf('Theta is\n');
disp(Theta1);
fprintf('Please hit enter to continue\n');
pause;
%% Part 5: Fit the first training example using weighted normal equation
figure;
hold on;
plot(lambda,firstexample,'rx','linewidth',2);
xlabel('lambda');
ylabel('intensity');
tau=5;
yhat=lwr(lambda,firstexample,tau);
plot(lambda,yhat);
hold off;
fprintf('Please hit enter to continue\n');
pause;
%% Part 6: Fit the first training example using weighted normal equation with different tau
figure;
hold on;
plot(lambda,firstexample,'rx','linewidth',2);
xlabel('lambda');
ylabel('intensity');
colors = {'r-', 'b-', 'g-', 'm-', 'c-'};
tau=[1 5 10 100 1000];
for i=1:length(tau)
    yhat=lwr(lambda,firstexample,tau(i));
    plot(lambda,yhat,colors{i});
end
legend('rawdata','tau=1','tau=5','tau=10','tau=100','tau=1000');
hold off;
fprintf('Please hit enter to continue\n');
pause;
%% Part 7: Smooth the training set and the test set
m=size(training,1);
m1=size(test,1);
training2=training;
test2=test;
for i=1:m
    smoothed=lwr(lambda,training(i,:),5);
    training2(i,:)=smoothed;
end
for k=1:m1
    smoothed1=lwr(lambda,test(k,:),5);
    test2(k,:)=smoothed1;
end
fprintf('Please hit enter to continue\n');
pause;
%% Part 8: Fitting the left part of the spectrum for training set:Functional regression
% Separate the training set and test set into left part and right part;
trainingleft=training2(:,1:50);
testleft=test2(:,1:50);
trainingright=training2(:,151:end);
testright=test2(:,151:end);
% Form the matrix for the functional distance
trainingdistance=zeros(m);
for i=1:m
    for j=1:i
        trainingdistance(i,j)=norm(trainingright(i,:)-trainingright(j,:))^2;
        trainingdistance(j,i)=trainingdistance(i,j);
    end
end
% For each index, find the minimum 3 distance
number=3;
trainingleftrevised=trainingleft;
for i=1:m
    [mini,minindex]=fishsort(trainingdistance(i,:),number);
    ker=max(0,1-mini/(max(trainingdistance(i,:))));
    trainingleftrevised(i,:)=ker'*trainingleft(minindex,:)/sum(ker);
end
difference=trainingleftrevised-trainingleft;
erroraverage=sum(difference(:).^2)/m;
fprintf('The average error for training set is:\n');
disp(erroraverage);
fprintf('Please hit enter to continue\n');
pause;
%% Part 9: Functional regression for test set
testdistance=zeros(m1); %Preallocation
for i=1:m1
    for j=1:m
        testdistance(i,j)=norm(testright(i,:)-trainingright(j,:))^2;
    end
end
testleftrevised=testleft;
for i=1:m1
    [mini,minindex]=fishsort(testdistance(i,:),number);
    ker=max(0,1-mini/(max(testdistance(i,:))));
    testleftrevised(i,:)=ker'*trainingleft(minindex,:)/sum(ker);
end
difference2=testleftrevised-testleft;
erroraverage2=sum(difference2(:).^2)/m1;
fprintf('The average error for test set is:\n');
disp(erroraverage2);
fprintf('Please hit enter to continue\n');
pause;
%% Part 10: Plot the revised spectrum for the test set 1 and 6
lambdaleft=lambda(1:50);
y1=test2(1,:);
y1_revised=testleftrevised(1,:);
y6=test2(6,:);
y6_revised=testleftrevised(6,:);
figure;
hold on;
plot(lambda,y1,'k-','linewidth',2);
plot(lambdaleft,y1_revised,'r-','linewidth',2);
legend('unrevised','revised');
hold off;
figure;
hold on;
plot(lambda,y6,'k-','linewidth',2);
plot(lambdaleft,y6_revised,'r-','linewidth',2);
legend('unrevised','revised');
hold off;