function [err, model, erTest] = problem1(K)
% This function is for problem 1 of HW1 assignment and takes one argument
% K.
% K is the highest degree of the polynomial for curve fitting.
% The function loads dataset1a, randomly divides the dataset into training
% and test sets and then calls polyreg()/

load dataset1a.txt;
X = dataset1a(:,1);
Y = dataset1a(:,2);
%intr stores 100 random indices between 1 and 200 to get the training set
%from the dataset
[indtr] = randperm(200,100)';
trainingset = [X(indtr),Y(indtr)];
% list of all indices
[indices] = 1:200; indices = indices';
%indt stores indices for the test set by checking for common indices
%between all 'indices' and the training set indices. 
[indt] = indices(~ismember(indices,indtr));
testset = [X(indt),Y(indt)];
x1 = trainingset(:,1);
y1 = trainingset(:,2);
x2 = testset(:,1);
y2 = testset(:,2);

p=1;
for j = 1:K+1
    
    [err, model, erTest] = polyreg(x1,y1,j,x2,y2)
    errors1(p) = err;
    errors2(p) = erTest;
    models = model;
    p=p+1;
end    
figure;
plot(errors1,'r');
hold on;
plot(errors2,'b');
legend('Training Risk', 'Testing Risk');
model = models;
q = min(X):(max(X)/70):max(X);
qq = zeros(length(q),K+1);

for i=1:K+1
  qq(:,i) = q.^(K+1-i);
end
    size(qq)
    size(model)
    figure;
    plot(q,(qq*model),'r-');
    hold on;
    plot(x2,y2, 'b*');
    title('Best Model');
    
    legend('Prediction', 'Actual points');