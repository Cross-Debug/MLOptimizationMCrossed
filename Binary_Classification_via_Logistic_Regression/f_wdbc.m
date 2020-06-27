function f = f_wdbc (wh,D,mu)
X = D(1:31,:);
y = D(32,:);
P = length(y);
f = 0;
for p = 1:P % This can be done with matrix operations for speed. 
    xhp = X(:,p);
    fp = log (1+exp(-y(p)*(wh'*xhp) ));
    f = f + fp;
end
f = f/P + .5*mu*(wh'*wh);
end

