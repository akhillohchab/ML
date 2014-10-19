function [err,model,errT] = polyreg_new(x,y,D,xT,yT)
%This is just a modification of the polyreg function, except the polynomial
%function here is different.
% 
% 
%   This is specifically for problem2 of HW1 assignment.
% 
% Finds a D-1 order polynomial fit to the data
%
%    function [err,model,errT] = polyreg(x,y,D,xT,yT)
%
% x = vector of input scalars for training set
% y = vector of output scalars for training set
% D = the order plus one of the polynomial being fit
% xT = vector of input scalars for testing set
% yT = vector of output scalars for testing set
% err = average squared loss on training
% model = vector of polynomial parameter coefficients
% errT = average squared loss on testing
%
%


xx = zeros(length(x),2*D+1);

xx(:,1) = 1;
for i=2:D+1
  xx(:,i) = sin(i*x);
end
for i=D+2:2*D+1
  xx(:,i) = cos(i*x);
end
model = pinv(xx)*y;
err   = (1/(2*length(x)))*sum((y-xx*model).^2);

if (nargin==5)
  xxT = zeros(length(xT),2*D+1);
  xxT(:,1) = 1;
  for i=2:D+1
    xxT(:,i) = sin(i*xT);
  end
  for i=D+2:2*D+1
    xxT(:,i)= cos(i*xT);
  end
  errT  = (1/(2*length(xT)))*sum((yT-xxT*model).^2);
end
figure;
plot(x,y,'X')
q  = (min(x):(max(x)/300):max(x))';
qq = zeros(length(q),2*D+1);
qq(:,1) = 1;
for i=2:D+1
  qq(:,i) = sin(i*q);
end

for i=D+2:2*D+1
  qq(:,i) = cos(i*q);
end

clf

plot(x,y,'X');
hold on
if (nargin==5)
plot(xT,yT,'cO');
end
plot(q,qq*model,'r')
legend('Training', 'Test', 'Model');

