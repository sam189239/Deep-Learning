X = csvread("train.csv");
y = (X(2:end,2))';  %1 * 891
X = X(2:end, [3,6,7,8,9,11]);  % 891 * 6 
X = X'; % 6 * 891
m = size(X,2);
n_x = size(X,1);
n_h = 6;
W1 = randn(n_h,size(X,1));  % 4 * 6
b1 = zeros(n_h,1);
W2 = randn(1,n_h);
b2 = zeros(1,1);
num_iters = 10000;
alpha = 0.005;

%figure1 = plot(X,y)



function [A2,A1,Z1,Z2] = fwd(X,W1,b1,W2,b2,y)
    m = size(X,2);
    Z1 = ((W1 * X) + b1);  % 4 * 891
    A1 = tanh(Z1);

    Z2 = (W2 * A1) + b2; % 1 * 1
    A2 = sigmoid(Z2);
   
end

function [dW1,db1,dW2,db2] = bwd(X,y,A1,A2,W2,W1)

    m = size(X,2);
    dZ2 = A2 - y;
    dW2 = 1/m * (dZ2 * A1');
    db2 = 1/m * sum(dZ2,axis =2);
    dZ1 = (W2' * dZ2) .* (1 - A1.^2);
    dW1 = 1/m * (dZ1 * X');
    db1 = 1/m * sum(dZ1,axis = 2); 

end

function [W1,b1,W2,b2] = update(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)

    W1 = W1 - alpha .* dW1;
    b1 = b1 - alpha .* db1;
    W2 = W2 - alpha .* dW2;
    b2 = b2 - alpha .* db2;

end

function cost = compute_cost(y,A2)
    
    m = size(y,2);
    cost = -1/m * sum(y .* log(A2)+(1-y) .* log(1-A2));

end

c = zeros(num_iters/100,1);
d = 100:100:num_iters;

for i = 1:num_iters
    [A2,A1,Z1,Z2] = fwd(X,W1,b1,W2,b2,y);
    cost = compute_cost(y,A2);
    [dW1,db1,dW2,db2] = bwd(X,y,A1,A2,W2,W1);
    [W1,b1,W2,b2] = update(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha);
    if mod(i,100) == 0
        fprintf("Cost after %d iterations = %.2f \n",i,cost);
        c(i/100,1) = cost;
    end   
end
 
plot(d,c);
xlabel("No. of Iterations");
ylabel("Cost");
title("ML Bois model - Titanic Survivors");


[A2,A1,Z1,Z2] = fwd(X,W1,b1,W2,b2,y);
p = A2>0.5;
acc = mean(double(p == y)) * 100;
fprintf("Accuracy = %f \n",acc);

fprintf("Testing...");
X_test = csvread("test.csv");
X_test = (X_test(2:end, [2,5,6,7,8,10]))';
y_test = csvread("gender_submission.csv");
y_test = (y_test(2:end,2))';

[A2,A1,Z1,Z2] = fwd(X_test,W1,b1,W2,b2,y_test);
p = A2>0.5;
acc = mean(double(p == y_test)) * 100;
fprintf("Accuracy in test set = %f \n",acc);






