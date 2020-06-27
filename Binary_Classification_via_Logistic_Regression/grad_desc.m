% To implement the gradient descent algorithm.
% Example: [xs,fs,k] = grad_desc('f_rosen','g_rosen',[0; 2],iterations,D,mu); ColumnV
% Version L3, MC. Edits to feval to include D, mu.
function [xs,fs,k] = grad_desc(fname,gname,x0,K,D,mu)
format compact
format long
k = 1;
xk = x0;
gk = feval(gname,xk,D,mu);
dk = -gk;
ak = bt_lsearch2019(xk,dk,fname,gname,D,mu);
adk = ak*dk;
while k <= K
  xk = xk + adk;
  gk = feval(gname,xk,D,mu);
  dk = -gk;
  ak = bt_lsearch2019(xk,dk,fname,gname,D,mu);
  adk = ak*dk;
  k = k + 1;
end
disp('solution:')
xs = xk + adk
disp('objective function at solution point:')
fs = feval(fname,xs,D,mu)
format short
disp('number of iterations performed:')
k