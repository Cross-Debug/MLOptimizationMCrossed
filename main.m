% Each car is 1 vertical vector. Accompnaying dataset is D_mpg.mat. Thank you to Statlib at the Carnegie Mellon University for the data.
% 6 parameters (Rows 1-6 are parameters of the vehicle.)
% This is a model that predicts fuel economy.

y = D_mpg(7,:)'; % Takes whole 7th row, transpose. These are the grand truths.
M = length (y); 
P = 314;
T = M-P;
X = D_mpg(1:6,:); % Takes all of rows 1:6
Xh = [X; ones(1,M)]; % Takes X and makes the last row vector of [1xM] all 1s

% Takes all rows and all samples up until 314, the learning set
Xh_tr = Xh(:,1:P); 
y_tr = y(1:P);

%Test Matrix
Xh_te = Xh(:,P+1:M);
y_te = y(P+1:M);

wh = (Xh_tr*Xh_tr') \ (Xh_tr*y_tr);
trainedResults = wh' * Xh_tr; % Output as row vector
% size( sum((trainedResults - y_tr)^2, 'all') )
RMSE_tr = (sum((trainedResults - y_tr').^2, 'all' ) / 314 )^(1/2);

testResults = wh' * Xh_te;
RMSE_te = (sum((testResults - y_te').^2, 'all' ) / 78 )^(1/2);


plotA = plot(testResults, 'red');
hold;
plotA = plot(y_te, 'blue');
title('Plot of GND Truth and Linear Model Prediction');
legend( 'Linear Model Prediction','Ground Truth');
xlabel('Sample Number');
ylabel('Fuel Economy [mpg]');
