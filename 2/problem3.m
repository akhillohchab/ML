function problem3()
% Function created for HW2, problem 3 that calculates the gram matrix for a
% given set of points and a given function
X = [0.25, 0.3, 0.4; 0.2, 0.4, 0.25; 0.3, 0.03, 0.2; 0.1, 0.2,0.15; 0.03, 0.05, 0.7];

y=[];
Gram = zeros(5,5);

for i=1:5
    for j=1:5
     y((i-1)*5+j,:) = dot(X(i,:),X(j,:));   
    Gram (i,j) = 1 - exp(-((y((i-1)*5+j,:)/0.75)^3));
    end
end

fprintf('Gram Matrix:\n\n');
Gram
        
      


end

