function [err, model] = logistic(X,y,e)
t =1;
[theta] = rand(1,3)
past_theta = [0 0 0];
while (true)
	n=0.5;
	f = X*(theta');
	g = 1/(1+exp(-f))
	fx = [g>0.5]';
	%class_error calculates the classification error by taking the
	%term-wise difference between fx and Y by counting the indices at which
	%the difference between the predicted value and the actual value
	%differ, i.e. the difference is not equal to zero.
    class_error = sum((fx-y)~=0)
    gradg = fx.*(1-fx);
    temp = (fx-y)'*X;
	grad_emp = temp;
%     empsize = size(grad_emp) 
%     emp_risk = (1/length(X))*(y-1).*log(1-g')-y.*log(g');   
    past_theta = theta
    theta = past_theta - (n*grad_emp);
	t=t+1;
%     size(theta)
%     size(theta_new)

    diff = abs(theta - past_theta);
% 	theta = theta_new;
    if (diff<= e)
        break;
    end
end

model = theta;
err = class_error;

