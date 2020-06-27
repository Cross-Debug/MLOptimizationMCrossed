load D_wdbc.mat
% Data Pull
Dtr = D_wdbc(:,1:285);
Dte = D_wdbc(:,286:569);
% Normalization
Xtr = zeros(30,285);
m = zeros(1,30);
v = zeros(1,30);
% Labels
ytr = Dtr(31,:);
yte = Dte(31,:);
% Norm tr
for k = 1:30
    xi = Dtr(k,:); % xi takes each row.
    m(k) = mean(xi);
    v(k) = sqrt(var(xi));
    Xtr(k,:) = (xi - m(k))/v(k);
end
normDtr = Xtr;
normDtr(31,1:285) = ones(1,285);
normDtr(32,1:285) = ytr;
% Normal te
Xte = zeros(30,284);
for k = 1:30
    xi = Dte(k,:);
    Xte(k,:) = (xi - m(k))/v(k);
end % We now have Xtr and Xte normalized sets.
normDte = Xte;
normDte(31,:) = ones(1,284); 
normDte(32,:) = yte; % We'll feed Dte = [ nums 1 label ] into GD.
% Modeling
wh_initial = zeros(1,31)';
mu = 0;
KK = 10; % Number of iterations.
[xs1,fs1,k1] = grad_desc('f_wdbc','g_wdbc',wh_initial,KK,normDtr,mu);
% Run2
mu = .1;
KK = 10;
[xs2,fs2,k2] = grad_desc('f_wdbc','g_wdbc',wh_initial,KK,normDtr,mu);
% Run3
mu = .0;
KK = 30;
[xs3,fs3,k3] = grad_desc('f_wdbc','g_wdbc',wh_initial,KK,normDtr,mu);
% Run4
mu = .075;
KK = 30;
[xs4,fs4,k4] = grad_desc('f_wdbc','g_wdbc',wh_initial,KK,normDtr,mu);
% Classification
signDte1 = xs1'*normDte(1:31,:);
signDte2 = xs2'*normDte(1:31,:);
signDte3 = xs3'*normDte(1:31,:);
signDte4 = xs4'*normDte(1:31,:);
% Confusion Matrix
C1 = zeros(2,2);
for k = 1:284
   if signDte1(k) > 0 && yte(k) == 1
       C1(1,1) = C1(1,1) + 1;
   elseif signDte1(k) < 0 && yte(k) == 1
       C1(2,1) = C1(2,1) + 1;
   elseif signDte1(k) < 0 && yte(k) == -1
       C1(2,2) = C1(2,2) + 1;
   elseif signDte1(k) > 0 && yte(k) == -1
       C1(1,2) = C1(1,2) + 1;
   else 
       print("Error");
   end
end

C2 = zeros(2,2);
for k = 1:284
   if signDte2(k) > 0 && yte(k) == 1
       C2(1,1) = C2(1,1) + 1;
   elseif signDte2(k) < 0 && yte(k) == 1
       C2(2,1) = C2(2,1) + 1;
   elseif signDte2(k) < 0 && yte(k) == -1
       C2(2,2) = C2(2,2) + 1;
   elseif signDte2(k) > 0 && yte(k) == -1
       C2(1,2) = C2(1,2) + 1;
   else 
       print("Error");
   end
end
    
C3 = zeros(2,2);
for k = 1:284
   if signDte3(k) > 0 && yte(k) == 1
       C3(1,1) = C3(1,1) + 1;
   elseif signDte3(k) < 0 && yte(k) == 1
       C3(2,1) = C3(2,1) + 1;
   elseif signDte3(k) < 0 && yte(k) == -1
       C3(2,2) = C3(2,2) + 1;
   elseif signDte3(k) > 0 && yte(k) == -1
       C3(1,2) = C3(1,2) + 1;
   else 
       print("Error");
   end
end

C4 = zeros(2,2);
for k = 1:284
   if signDte4(k) > 0 && yte(k) == 1
       C4(1,1) = C4(1,1) + 1;
   elseif signDte4(k) < 0 && yte(k) == 1
       C4(2,1) = C4(2,1) + 1;
   elseif signDte4(k) < 0 && yte(k) == -1
       C4(2,2) = C4(2,2) + 1;
   elseif signDte4(k) > 0 && yte(k) == -1
       C4(1,2) = C4(1,2) + 1;
   else 
       print("Error");
   end
end
% Accuracy
acc_te1 = sum(diag(C1))/284*100
acc_te2 = sum(diag(C2))/284*100
acc_te3 = sum(diag(C3))/284*100
acc_te4 = sum(diag(C4))/284*100