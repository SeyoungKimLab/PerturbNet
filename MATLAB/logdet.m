function [ld] = logdet(A)
    spA = sparse(A);
    [Lq, p] = chol(spA, 'lower');
    %assert(p == 0);
    if p ~= 0
        %warning('Taking determinant of non-PD matrix');
        ld = -inf;
        return;
    end
    ld = 2*sum(log(full(diag(Lq))));
end
