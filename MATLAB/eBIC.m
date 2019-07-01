function [bic_gamma] = eBIC(Y, X, Lambda, Theta, gamma)
    ll = loglik(Y, X, Lambda, Theta);
    Ecard = nnz(triu(Lambda,1)) + nnz(Theta);
    [n, q] = size(Y);
    [n, p] = size(X);
    bic_gamma = -2*ll + Ecard*log(n) + 4*Ecard*gamma*log(p+q);
end
