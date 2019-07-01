function [b] = BIC(Y, X, Lambda, Theta)
    ll = loglik(Y, X, Lambda, Theta);
    Ecard = nnz(triu(Lambda,0)) + nnz(Theta);
    [n, q] = size(Y);
    [n, p] = size(X);
    b = -2*ll + Ecard*log(n);
end
