function [error, model, errorTest] = problem2(K)
% problem2 is a function specifically for problem2 of HW1 assignment that
% takes one argument : K
% K is the highest degree of parameter allowed.
% 
% The function loads the dataset1b, divides it into training and testing 
% sets and calls the polynomial regression function polyreg_new with 5 
% arguments. 
% Note that the argument passed to polyreg_new is K+1 and not K.

load dataset1b.txt;
X = dataset1b(:,1);
Y = dataset1b(:,2);
trainingset = [X(1:100),Y(1:100)];
testset = [X(101:200),Y(101:200)];
x1 = trainingset(:,1);
y1 = trainingset(:,2);
x2 = testset(:,1);
y2 = testset(:,2);


for j = 1:K+1
    
    [err(j), model, erTest(j)] = polyreg_new(x1,y1,j,x2,y2);
    
end    

% Plots of Training vs Testing error.
figure;
plot (1:K+1,err, 'r-');
hold on;
plot (1:K+1,erTest, 'b-');
legend('Training error', 'Testing error');
hold off;
index = find(erTest==min(erTest));
error = err(index);
errorTest = erTest(index);
