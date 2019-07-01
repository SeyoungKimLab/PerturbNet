function [ll] = loglik(Y, X, Lambda, Theta)
% l_n for GGM: (see Foygel & Drton)
%     n/2 * (logdet(Lambda) - tr(S*Lambda))
% l_n for CGGM
%     n/2 * (logdet(Lambda) - tr(Syy Lambda + 2Sxy^T*Theta + Lambda^-1 Theta^T Sxx Theta))

    cov_all = cov([Y X]);
    [n, p] = size(X);
    [n, q] = size(Y);
    Syy = cov_all(1:q,1:q);
    Sxy = cov_all(q+1:end,1:q);
    Sxx = cov_all(q+1:end,q+1:end);

    tr_1 = trProd(Syy, Lambda);
    tr_2 = 2*trProd(Sxy,Theta);
    tr_3 = trProd((Lambda\(Theta'))', Sxx*Theta);
    tr_term = tr_1 + tr_2 + tr_3;
    ll = 0.5 * n * (logdet(Lambda) - tr_term);
end

function [t] = trProd(A, B)
% returns tr(A^T B)
    t = full(sum(sum(A .* B)));
end
