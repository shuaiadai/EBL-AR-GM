%% generate the dataset (channels are independent)

m = 5; % number of nodes
N = 1000; % sample size 
e = randn(N,m);
y = zeros(N,m);
for k = 3:N
    y(k,:) = -0.1*y(k-1,:)-0.6*y(k-2,:)+e(k,:);
end

%% Indentification of a Sparse Graphical Model of order 2

[Omega,S] = identS(y,2);

