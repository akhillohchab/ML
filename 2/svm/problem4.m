function [] = problem4()
% Function takes no arguments and was created for HW2, problem 4.
% To build a SVM classifier for the given dataset
% 
% The function svc.m has been modified to include another output, the
% maximum margin width.
% 

% % Linear Kernels

global p1;
global p2;
% the array C is initialized which will be used by linear, poly and rbf
% kernels

C = [1e-6;1e-5;1e-4;1e-3;1e-2;1e-1;1;10;100;1000;10000;100000] ; 

load 'svm-dataset.mat';

% Calculate random 50 indices to divide the dataset into training set and
% test set of equal sizes
indices = 1:100;
[training_ind1] = randperm(100,50);

test_ind1 = indices(~ismember(indices,training_ind1));
% training set
x = X(training_ind1,:);
y = Y(training_ind1);
% test set
x2 = X(test_ind1,:);
y2 = Y(test_ind1);

% initializing the different storage containers/arrays
number_SV = [];
sum_alphas = [];
tested_C = [];
errors = [];
margins = [];
store = zeros(12,4);
fprintf('Linear Kernels:\n\n');
i=1;
for l=1:length(C)

    [num alpha b max_margin] = svc(x,y,'linear', C(l));
    
    err = svcerror(x,y,x2,y2,'linear',alpha, b)/length(y2);
    
    number_SV = [number_SV;num];
    sum_alphas = [sum_alphas;sum(alpha)];
    tested_C = [tested_C; C(l)];
    errors = [errors;err];
    margins = [margins; max_margin];
    store(i,:) = [err max_margin num sum(alpha)];
    i=i+1;
end
% fprintf('\n\nResults for Linear Kernels:\n\nValue of C\tNo. of SV\tSum of alphas\n');
% horzcat(tested_C,number_SV, sum_alphas,errors, margins)
store = [tested_C store ];
save('LinearKernels_Table.mat','store') ;

%  Plotting the error against C. log C taken for easy reading
figure;
% Plot error vs logC for linear kernels
plot(log10(C), store(:,2), 'r-');
xlabel('Log10 (C)');
ylabel('Classification error');
title('Linear Error');

figure;
% Plot margin vs logC for Linear Kernels
plot(log10(C), log10(store(:,3)), 'b-');
xlabel('Log10 (C)');
ylabel('log(Maximum Margin)');
title('Linear margin');

%% Polynomial Kernels
fprintf('\nPolynomial Kernels:\n\n');
% initializing different storage containers/arrays
store_poly_error = zeros(12,5);
store_poly_margins = zeros(12,5);
store_poly_SV = zeros(12,5);
store_poly_sum = zeros(12,5);


for k = 1:5
% the value of global variable p1 is used to determine the degree of the
% polynomial in the polynomial kernel in question
p1 = k;
number_SV = [];
sum_alphas = [];
tested_C = [];
errors = [];
margins = [];


    for l = 1:length(C)
%     50 random indices are generated again to divide the dataset into
%     training set and test set of equal sizes. This was done again to try
%     and get different training/test sets for different value of C
    [training_ind1] = randperm(100,50);

    test_ind1 = indices(~ismember(indices,training_ind1));

    x = X(training_ind1,:);
    y = Y(training_ind1);

    x2 = X(test_ind1,:);
    y2 = Y(test_ind1);

        
    [num, alpha, b, max_margin] = svc(x,y,'poly', C(l));
    
    err = svcerror(x,y,x2,y2,'poly',alpha, b)/length(y2);
    
    number_SV = [number_SV;num];
    sum_alphas = [sum_alphas;sum(alpha)];
    tested_C = [tested_C; C(l)];
    errors = [errors;err];
    margins = [margins; max_margin];
    store_poly_error(l,k) = err;
    store_poly_margins(l,k) = log10(max_margin);
    store_poly_SV (l,k) = num;
    store_poly_sum (l,k)= sum(alpha);

    
    end
   
% fprintf('\n\nResults for polynomial Kernels of degree%d', p1);
% fprintf('\n\nValue of C\tNo. of SV\tSum of alphas\n');
% horzcat(tested_C,number_SV, sum_alphas,errors)

end
% store_poly_error = [tested_C store_poly_error];
save('PolynomialKernels_errors.mat','store_poly_error') ;
% store_poly_error = [tested_C store_poly_margins];

save('PolynomialKernels_margins.mat','store_poly_margins') ;
% store_poly_error = [tested_C store_poly_SV];

save('PolynomialKernels_SV.mat','store_poly_SV') ;
% store_poly_error = [tested_C store_poly_sum];

save('PolynomialKernels_SumAlphas.mat','store_poly_sum') ;

% % 
% % Plotting Polynomial Kernels
% % 
% 
% 


% set(gcf,'numbertitle','off','name','Polynomial Kernel error')
figure;
hold on;
plot(log10(C), store_poly_error(:,1),'color' , [0 0 1]);
xlabel('Log10 (C)');
ylabel('Classification error');
plot(log10(C), store_poly_error(:,2), 'color' , [0 1 0]);
plot(log10(C), store_poly_error(:,3), 'color' , [1 0 0]);
plot(log10(C), store_poly_error(:,4), 'color' , [1 0 1]);
plot(log10(C), store_poly_error(:,5), 'color' , [0 1 1]);
% plot(log10(C), store_poly_error(:,1), 'color' , [0 1 1]);

legend('d=1','d=2','d=3','d=4','d=5');
title('Polynomial Error');

hold off;

figure;
hold on;
plot(log10(C), log10(store_poly_margins(:,1)),'color', [0 0 1]);
xlabel('Log10 (C)');
ylabel('log(Maximum Margin)');
plot(log10(C), log10(store_poly_margins(:,2)),'color', [1 0 0]);
plot(log10(C), log10(store_poly_margins(:,3)), 'color' , [0 1 1]);
plot(log10(C), log10(store_poly_margins(:,4)), 'color' , [0 1 0]);
plot(log10(C), log10(store_poly_margins(:,5)), 'color' , [1 0 1]);
% 
legend('d=1','d=2','d=3','d=4','d=5','location','northeast');
title('Polynomial Margin');

hold off;

%% RBF Kernels

%  initializing the sigma array, this will be used for rbf kernels
Sigma = [1e-6;1e-5;1e-4;1e-3;1e-2;1e-1;1;10;100;1000;10000;100000] ;

fprintf('\nRBF Kernels:\n\n');
store_rbf_error = zeros(12,5);
store_rbf_margins = zeros(12,5);
store_rbf_SV = zeros(12,5);
store_rbf_sum = zeros(12,5);


for k = 1:length(Sigma)
p1 = Sigma(k);
number_SV = [];
sum_alphas = [];
tested_C = [];
errors = [];
margins = [];

    for l = 1:length(C)    
        
    [training_ind1] = randperm(100,50);

    test_ind1 = indices(~ismember(indices,training_ind1));

    x = X(training_ind1,:);
    y = Y(training_ind1);

    x2 = X(test_ind1,:);
    y2 = Y(test_ind1);

    [num, alpha, b0, max_margin] = svc(x,y,'rbf', C(l));
    
    err = svcerror(x,y,x2,y2,'rbf',alpha, b0)/length(y2);
    
    number_SV = [number_SV;num];
    sum_alphas = [sum_alphas;sum(alpha)];
    tested_C = [tested_C; C(l)];
    errors = [errors;err];
    margins = [margins; max_margin];
    store_rbf_error(l,k) = err;
    store_rbf_margins(l,k) = max_margin;
    store_rbf_SV (l,k) = num;
    store_rbf_sum (l,k)= sum(alpha);

    
    end
   

end
store_rbf_error = [tested_C store_rbf_error];
save('RbfKernels_errors.mat','store_rbf_error') ;
save('RbfKernels_margins.mat','store_rbf_margins') ;
save('RbfKernels_SV.mat','store_rbf_SV') ;
save('RbfKernels_SumAlphas.mat','store_rbf_sum') ;

% Plot Rbf Kernels

% set(gcf,'numbertitle','off','name','Polynomial Kernel error')
figure;
hold on;
plot(log10(C), store_rbf_error(:,6),'color' , [0 0 1]);
xlabel('Log10 (C)');
ylabel('Classification error');
plot(log10(C), store_rbf_error(:,8), 'color' , [0 1 0]);
plot(log10(C), store_rbf_error(:,9), 'color' , [1 0 0]);
plot(log10(C), store_rbf_error(:,10), 'color' , [1 0 1]);
plot(log10(C), store_rbf_error(:,12), 'color' , [0 1 1]);
title('RBF Error');

legend('sigma 0.1','sigma 10','sigma 100','sigma 1000','sigma 10000','location', 'Northeast');
hold off;

figure;
hold on;
plot(log10(C), log10(store_rbf_margins(:,6)),'color', [0 0 1]);
xlabel('Log10 (C)');
ylabel('log(Maximum Margin)');
plot(log10(C), log10(store_rbf_margins(:,8)),'color', [1 0 0]);
plot(log10(C), log10(store_rbf_margins(:,9)), 'color' , [0 1 1]);
plot(log10(C), log10(store_rbf_margins(:,10)), 'color' , [0 1 0]);
% plot(log10(C), log10(store_rbf_margins(:,12)), 'color' , [1 0 1]);
% plot(log10(C), log10(store_rbf_margins(:,5)), 'color' , [1 1 0]);

% 
legend('sigma 0.1','sigma 10','sigma 100','sigma 1000', 'location', 'Northeast');
title('RBF Margin');
hold off;



end



