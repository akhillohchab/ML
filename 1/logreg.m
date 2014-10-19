function [err,model] = polyreg(x,y,e)
%
% Finds a D-1 order polynomial fit to the data
%
%    function [err,model,errT] = polyreg(x,y,D,xT,yT)
%
% x = vector of input scalars for training
% y = vector of output scalars for training
% D = the order plus one of the polynomial being fit
% xT = vector of input scalars for testing
% yT = vector of output scalars for testing
% err = average squared loss on training
% model = vector of polynomial parameter coefficients
% errT = average squared loss on testing
%
% Example Usage:
%
% x = 3*(rand(50,1)-0.5);
% y = x.*x.*x-x+rand(size(x));
% [err,model] = polyreg(x,y,4);
%

% clf
% plot(x,y,'X');
% hold on
% if (nargin==5)
% plot(xT,yT,'cO');
% end
% plot(q,qq*model,'r')
% 
D = size(x,2);
size(x)
t=1;
% xx = zeros(size(x,1),D+1);
% xx(:,1) = 1;
% for i=2:D+1
%     xx(:,i) = x[:,i-1];
% end
theta0 = [(max(x)+min(x))/2];
n=1/t;

[f] = x*theta0';
[g]= [1/(1+exp(-f))]';
[gradg] = g.*(1-g);
[pred] =[g>0.5];
pred = [y-pred];
errors =sum(pred==0);

grad_emp = (1/length(x))*sum(((1-y)/(1-f))-(y/f))*gradg;



theta = zeros(1000, D);

theta(1,:) = [theta0-grad_emp];
i=1;

diff = norm(theta(1,:)) - norm(theta0);
while(diff>e)
    [f] = X*theta(i,:)';
    g= (1+exp(-f).^(-1));
    [gradg] = g.*(1-g);
    [pred] = [g>0.5];
    errors =sum(((y - pred)==0)==0)
    grad_emp = (1/size(x,1))*sum(((1-y)/(1-f))-(y/(f)))*gradg
    t=t+1;
    n=1/t;
    theta(i+1,:) = theta(i,:) - n*grademp
    diff = norm(theta(i+1,:)) - norm(theta(i,:));
    i=i+1;
    
end
theta = theta(1:i,:);    
model = theta(i,:);
f = x*model';
g = [(1/(1+exp(-f)))>0.5]';
   err = sum(((y-g)==0)==0)
   scatter3(x(:,1),x(:,2),g)
    
  
    
