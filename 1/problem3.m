function [err, model, erTest] = problem3(e,n)
% problem3 is the function specifically for problem3 of HW1 and takes two
% arguments : e,n and returns two values : classification error and the
% learned model
% e is the tolerance value
% n is the step size for gradient descent.
% 
% 
% this function loads dataset2 and passes X,Y to the logistic regression
% function called myfunc along with the e,n arguments.
load dataset2;

[error, model ] = myfunc(X,Y,e,n);

error 
model