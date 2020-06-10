load X_iris.mat

X1 = X_iris(:,1:50);
Xtr1 = X1(:,1:35);
Xte1 = X1(:,36:50);
X2 = X_iris(:,51:100);
Xtr2 = X2(:,1:35);
Xte2 = X2(:,36:50);
X3 = X_iris(:,101:150);
Xtr3 = X3(:,1:35);
Xte3 = X3(:,36:50);

% Begin tr data
k = 5

% Classification 1
trainxp = [ Xtr1 Xtr2 Xtr3 ];
train1yp = [ ones(1,35) -ones(1,70) ];
[w1,C2_1] = LRBC_newton(trainxp,train1yp,k);
% Classification 2
train2yp = [ -ones(1,35) ones(1,35) -ones(1,35) ];
[w2,C2_2] = LRBC_newton(trainxp,train2yp,k);
% Classification 3
train3yp = [ -ones(1,70) ones(1,35) ];
[w3,C2_3] = LRBC_newton(trainxp,train3yp,k);

% Normalization

w1norm = sqrt( sum(w1(1:4).*w1(1:4)) );
w1 = w1./w1norm; % we contains w_hat and bias
w2norm = sqrt( sum(w2(1:4).*w2(1:4)) );
w2 = w2./w2norm;
w3norm = sqrt( sum(w3(1:4).*w3(1:4)) );
w3 = w3./w3norm;

% Begin te data

% SingleTest logic data
%singleTester3 = Xte3(1:4)';
%singleTester = [ 5.4 3.4 1.5 .4 ]';
%singleTester3'*w1(1:4)
%singleTester3'*w2(1:4)
%singleTester3'*w3(1:4)
%Correct, produces the a positive or large number for the correct class.

testxp = [ Xte1 Xte2 Xte3 ; ones(1,45) ];
what = [ w1 w2 w3 ];
classification = testxp' * what;

% Confusion matrix (Test data) Only works with set demensions.

C3 = zeros(3,3);
for k=1:15
    if classification(k,1) > classification(k,2)
        C3 = C3 + [ 1 0 0; 0 0 0;0 0 0];
    elseif classification(k,2) > classification(k,3)
        C3 = C3 + [ 0 1 0; 0 0 0;0 0 0];
    else
        C3 = C3 + [ 0 0 1; 0 0 0;0 0 0];
    end
end
for k=15:30
    if classification(k,1) > classification(k,2)
        C3 = C3 + [ 0 0 0; 1 0 0;0 0 0];
    elseif classification(k,2) > classification(k,3)
        C3 = C3 + [ 0 0 0; 0 1 0;0 0 0];
    else
        C3 = C3 + [ 0 0 0; 0 0 1;0 0 0];
    end
end
for k=30:45
    if classification(k,1) > classification(k,2)
        C3 = C3 + [ 0 0 0; 0 0 0;1 0 0];
    elseif classification(k,2) > classification(k,3)
        C3 = C3 + [ 0 0 0; 0 0 0;0 1 0];
    else
        C3 = C3 + [ 0 0 0; 0 0 0;0 0 1];
    end
end

sumCii = sum(diag(C3));
sumCij = sum(C3,'All');
classification_accuracy_test = sumCii/sumCij*100

% Confusion matrix for training data
trainxp = [ Xtr1 Xtr2 Xtr3 ; ones(1,105) ];

classificationt = trainxp' * what;
Ct = zeros(3,3);
for k=1:35
    if classificationt(k,1) > classificationt(k,2)
        Ct = Ct + [ 1 0 0; 0 0 0;0 0 0];
    elseif classificationt(k,2) > classificationt(k,3)
        Ct = Ct + [ 0 1 0; 0 0 0;0 0 0];
    else
        Ct = Ct + [ 0 0 1; 0 0 0;0 0 0];
    end
end
for k=36:70
    if classificationt(k,1) > classificationt(k,2)
        Ct = Ct + [ 0 0 0; 1 0 0;0 0 0];
    elseif classificationt(k,2) > classificationt(k,3)
        Ct = Ct + [ 0 0 0; 0 1 0;0 0 0];
    else
        Ct = Ct + [ 0 0 0; 0 0 1;0 0 0];
    end
end
for k=71:105
    if classificationt(k,1) > classificationt(k,2)
        Ct = Ct + [ 0 0 0; 0 0 0;1 0 0];
    elseif classificationt(k,2) > classificationt(k,3)
        Ct = Ct + [ 0 0 0; 0 0 0;0 1 0];
    else
        Ct = Ct + [ 0 0 0; 0 0 0;0 0 1];
    end
end

sumCii = sum(diag(Ct)); % This reuses variables
sumCij = sum(Ct,'All');
classification_accuracy_train = sumCii/sumCij*100

