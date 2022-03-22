function [Y, A, B, obj] = S3LRR13v2(XL, XU, YL, s_rank, lambda, inW0)
% inputs:
%   XL -- the labeled samples, format: d-by-ntr
%   XU -- the unlabeled samples, format: d-by-ntt
%   YL -- the label matrix corresponding to XL, format: ntr-by-c
%   s -- the low-rank parameter
%   lambda -- the regularization parameter
%
% outputs:
%   YU -- the estimated label matrix corresponding to XU, format: ntt-by-c
%   A -- discriminative subspace, format: d-by-s
%   B -- regression projection, format: s-by-c
%   obj -- objective function values
%   
% written by Yong Peng @ Hangzhou Dianzi University
% email: yongpeng@hdu.edu.cn
% October, 2021

% related parameters
[d, ntr] = size(XL);
[~, ntt] = size(XU);
[~, c] = size(YL);
n = ntr + ntt;

% centering the data matrix 
X = [XL, XU]; 
% X = mapminmax(X, -1, 1);
% H = eye(n) - ones(n)/n;
% X = X * H;
XL = X(:,1:ntr);

if nargin == 6
    % initialize 'D2' and 'YU'
    W = inW0;
    Wi2 = sqrt(sum(W.*W,2) + eps);
    d2 = 0.5./(Wi2);
    D2 = diag(d2);
else
    inW0 = (XL*XL' + 0.001*eye(d))\(XL*YL);
    W = inW0;
    Wi2 = sqrt(sum(W.*W,2) + eps);
    d2 = 0.5./(Wi2);
    D2 = diag(d2);
end

Y = zeros(n,c);
Y(1:ntr,:) = YL;
XHX = X * X';

maxIter = 2;
for t = 1:maxIter
%     fprintf('processing iteration %d...\n', t);
    
    % update 'Yu'
    for i = ntr+1:n
        tmp = X(:,i)'*W;
        Y(i,:) = EProjSimplex_new(tmp);
    end
    clear tmp
    %    normalize 'Y'
%     for j = 1:c
%         Y(:, j) = Y(:,j)./sqrt(sum(Y(:,j)));
%     end
    
    % update 'A'
    St = XHX + lambda*D2;
    Sb = X*(Y*Y')*X';
    [V,S] = eig(St\Sb);
    [~, idx_sorted] = sort(diag(S),'descend');
    A = V(:,idx_sorted(1:s_rank));
   
    % update 'B'
    XY = X * Y;
    B = (A'*St*A)\(A'*XY);
    
    % update 'D2'
    W = A*B;
    Wi2 = sqrt(sum(W.*W, 2)+eps);
    d2 = 0.5./Wi2;
    D2 = diag(d2);    
    
    % obj
    obj(t) = norm(Y - X'*W,'fro')^2 + lambda*sum(Wi2);
    
    if t>1
        diff = obj(t-1) - obj(t);
        if diff < 1e-5
            break
        end
    end
end
