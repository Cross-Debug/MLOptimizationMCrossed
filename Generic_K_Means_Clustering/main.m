% This program is for generic K means clustering. 
% Number of iterations = t. Number of classes = K.
% Data set = D, where each sample is a column vector.
% mu are the initial centers as column vectors.
% Scripted by Mike C.
% Data entry
D = [0 1 -1 2 -2 3;0 2 -1 3 1 1]; 
mu = [-1 2; 3 0];

% Parameters
K = 2;
Dlength = length(D);
muheight = size(mu,1)
r = zeros(Dlength,K);
distR = zeros(Dlength,K);
t = 2; % How many iterations you want.

for w = 1:t
% Calc dist
for m = 1:Dlength
    for l = 1:K
    distR(m,l) = norm(D(:,m)-mu(:,l));
    end
end
% Calc r matrix
r = zeros(Dlength,K);
[M,indexL] = max(distR,[],2);
for m = 1:Dlength
    r(m,indexL(m)) = r(m,indexL(m))+1; 
end
sumC = zeros(muheight,K);
countC = zeros(K,1);
for m = 1:Dlength
    sumC(:,indexL(m)) = sumC(:,indexL(m)) + D(:,m);
    countC(indexL(m),1) = countC(indexL(m),1) + 1; % CountC is column vector 
end
for m = 1:K
mu(:,m) = sumC(:,m)./countC(m,1);
end
end
% Plotting
scatter(D(1,:),D(2,:))
hold on
scatter(mu(1,:),mu(2,:))
legend('Data Points','Centers')
hold off
