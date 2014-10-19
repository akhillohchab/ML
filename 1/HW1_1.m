load dataset1a.txt
x = dataset1a(:,1);
y = dataset1a(:,2);

indices = randperm(200,100)

[trainingset] = [x(indices), y(indices)]
mind = 1:200
mind = mind'
[indt] = [mind(~ismember(mind, indices))]
[testset] = [x(indt), y(indt)]