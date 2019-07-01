function [a] = AIC(Y, X, Lambda, Theta)
    ll = loglik(Y, X, Lambda, Theta);
    Ecard = nnz(triu(Lambda,0)) + nnz(Theta);
    a = -2*ll + Ecard*2;
end
