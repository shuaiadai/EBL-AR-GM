
%% Install mex
mex projectSortC.c

%% generate the dataset (manifest variables are conditionally independent, #latent variables=2)

m = 5; % number of nodes
N = 1000; % sample size 
e = randn(N,m);
y = zeros(N,m);
for k=3:N
    y(k,:) = -0.1*y(k-1,:)-0.6*y(k-2,:)+e(k,:)+0.3*randn(1,2)*ones(2,m);
end

%% Indentification of a Latent-variable Graphical Model of order 2

[Omega,h,S,L] = identSL(y,2);
