X = csvread("train.csv");
y = (X(2:end,2))';  %1 * 891
X = X(2:end, [3,6,7,8,9,11]);  % 891 * 6 
X = X'; % 6 * 891
m = size(X,2);
n_x = size(X,1);
n_h1 = 20;
n_h2 = 20;
W1 = rand(n_h1,size(X,1))* (sqrt(1./m)) ;  % 4 * 6
b1 = zeros(n_h1,1);
W2 = rand(n_h2,n_h1)* (sqrt(1./n_h1));
b2 = zeros(n_h2,1);
W3 = rand(1,n_h2) * (sqrt(1./n_h2));
b3 = zeros(1,1);
num_iters = 20000;
alpha = 0.005;

%figure1 = plot(X,y)
function r = relu (z)
    r = max (0, z);
end

function ff = sigmoid(z)
    ff = 1./(1+exp(-z));
end



function [A3,A2,A1,Z1,Z2,Z3] = fwd(X,W1,b1,W2,b2,W3,b3,y)
    m = size(X,2);
    Z1 = ((W1 * X) + b1);  % 4 * 891
    A1 = relu(Z1);

    Z2 = (W2 * A1) + b2; %s
    A2 = relu(Z2);
    Z3 = (W3*A2)+ b3;
    A3 = sigmoid(Z3);
   
end

function [dW1,db1,dW2,db2,dW3,db3] = bwd(X,y,A1,A2,A3,W1,W2,W3)

    m = size(X,2);
    dZ3 = A3 - y;
    dW3 = 1/m * (dZ3 * A2');
    db3 = 1/m * sum(dZ3,axis =2);
    dZ2 = (W3' * dZ3) .* (1 - A2.^2);
    dW2 = 1/m * (dZ2 * A1');
    db2 = 1/m * sum(dZ2,axis = 2);
    dZ1 = (W2' * dZ2) .* (1- A1.^2);
    dW1 = 1/m * (dZ1 * X');
    db1 = 1/m * sum(dZ1,axis=2);

end

function [W1,b1,W2,b2,W3,b3] = update(W1,b1,W2,b2,W3,b3,dW1,db1,dW2,db2,dW3,db3,alpha)

    W1 = W1 - alpha .* dW1;
    b1 = b1 - alpha .* db1;
    W2 = W2 - alpha .* dW2;
    b2 = b2 - alpha .* db2;
    W3 = W3 - alpha .* dW3;
    b3 = b3 - alpha .* db3;

end

function cost = compute_cost(y,A3)
    
    m = size(y,2);
    cost = -1/m * sum(y .* log(A3)+ (1-y) .* log(1-A3));

end

c = zeros(num_iters/100,1);
d = 100:100:num_iters;

for i = 1:num_iters
    [A3,A2,A1,Z1,Z2,Z3] = fwd(X,W1,b1,W2,b2,W3,b3,y);
    cost = compute_cost(y,A3);
    [dW1,db1,dW2,db2,dW3,db3] = bwd(X,y,A1,A2,A3,W1,W2,W3);
    [W1,b1,W2,b2,W3,b3] = update(W1,b1,W2,b2,W3,b3,dW1,db1,dW2,db2,dW3,db3,alpha);
    if mod(i,100) == 0
        fprintf("Cost after %d iterations = %.2f \n",i,cost);
        c(i/100,1) = cost;
    end   
end
 
plot(d,c);
xlabel("No. of Iterations");
ylabel("Cost");
title("ML Bois model - Titanic Survivors");


[A3,A2,A1,Z1,Z2,Z3] = fwd(X,W1,b1,W2,b2,W3,b3,y);
p = A3>0.5;
%p = max(A3, [], 1);

acc = mean(double(p == y)) * 100;
fprintf("Accuracy = %f \n",acc);

fprintf("Testing...");
X_test = csvread("test.csv");
X_test = (X_test(2:end, [2,5,6,7,8,10]))';
y_test = csvread("gender_submission.csv");
y_test = (y_test(2:end,2))';

[A3,A2,A1,Z1,Z2,Z3] = fwd(X_test,W1,b1,W2,b2,W3,b3,y_test);
p = A3>0.5;
acc = mean(double(p == y_test)) * 100;
fprintf("Accuracy in test set = %f \n",acc);






