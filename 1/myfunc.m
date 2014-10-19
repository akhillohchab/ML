function [err, model] = myfunc(X,y,e,n)
% myfunc is a logistic regression function that takes 4 arguments: X,y, e,n
% 
% myfunc is for Problem 3 of HW1 assignment.
% 
% X = is the input part of the dataset
% y = is the classifier output
% e is the tolerance value
% n is the step size for gradient descent


% We initialize theta randomly and keep a variable 'past_theta' to keep the
% value of theta in the previous iteration. This is initialized as zero.
t=1;
[theta] = rand(1,3);
past_theta = [0 0 0];


% name of figure to plot classification error.
plterrors = figure;

while (true)

    %the  next three lines implement f(x;\theta) = (1+exp(-f))^(-1) and 
%     then implement a classifier by choosing a threshold as 0.5
    
    f = X*(theta');
	g = (1+exp(-f)).^(-1);
	fxt = [g>0.5];
	
    %term-wise difference between fx and Y by counting the indices at which
	%the difference between the predicted value and the actual value
	%differ, i.e. the difference is not equal to zero.
    tem = sum(abs(fxt-y));
    figure(plterrors);
    plot(t,tem,'b-');
    hold on;
    legend('Classification error');
    %In order to plot the empirical risk, we have to approximate the values
    %for fxt, otherwise Matlab would give NaN for different case
%     lnfx = ones(size(fxt));
%     lnfx (fxt==1) = 0.99;
%     
%     Empirical_risk(t) = (1/length(X))*([y-1].*(log(1-lnfx))-y.*(log(lnfx)));
    gradg = fxt.*(1-fxt);
    temp = (fxt-y)'*X;
	grad_emp = (1/length(X))*temp;
    past_theta = theta;
    theta = past_theta - (n*grad_emp);
	
    t=t+1;
    % This is where we check whether we should continue or not, by comparing
    % the difference value with the tolerance value     
    diff = abs(theta - past_theta);
    if (diff<= e)
        break;
    end
end
hold off;
model = theta;
% To get the plot of the 2D decision boundary, we first plot the points for
% which we get our predictor's value as 1 and 0, respectively. We can get
% the input parameters for these predictions by checking what indices for
% X1 and X2 give the output of the classifier as 1 and which ones give 0.
posindices = [y==1];
negindices = [y==0];
err = tem;
figure;

% Here we are just declaring the scale on both the axes for the plot
axis([min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2)) ]);
plot( X(posindices,1), X(posindices,2),'r+');
% scatter3(X(:,1),X(:,2),fx);
% view(0,90);
hold on
plot(X(negindices,1),X(negindices,2), 'b^');
hold on
% To plot the decision boundary, we make use of the fact that the
% decision boundary is a hyperplane given by: ?^T*X=0. So we can plot a
% line in 2D. To construct a line, we only need 2 points as a line can be
% constructed if two points are available.
x_endpoints = [min(X(:,1)), max(X(:,1))];

% To plot y, we use f = X*(theta'), and assume one of the parameters to be
% multiplied with y (where y=mx +c) is the equation of the line and the
% intercept c is also represented by a parameter value.
% Rearranging factors, we get:
 y_points = (-1./model(2)).*(model(1).*x_endpoints + model(3));

 plot(x_endpoints, y_points, 'k');
legend('Classifier Output: 1', 'Classifier Output: 2', 'Decision Boundary');
hold off
