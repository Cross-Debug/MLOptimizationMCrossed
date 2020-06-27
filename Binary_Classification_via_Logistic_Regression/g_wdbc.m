function g = g_wdbc (wh,D,mu)
X = D(1:31,:);
y = D(32,:); % The augmentation is done in main.m
P = length(y); 
f = zeros(31,1);
for p = 1:P
    xhp = X(:,p);
    fp = y(p)*xhp / (1+exp(y(p)*(wh'*xhp) )); % Mind that it's x_hat.
    f = f + fp;
end
g = -f/P + mu*wh;
end
