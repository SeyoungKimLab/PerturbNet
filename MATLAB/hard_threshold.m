function [tA] = hard_threshold(A, t)
    tA = A;
    find_A = find(A);
    find_tA = find(abs(A) >= t);
    victim_tA = setdiff(find_A, find_tA);
    tA(victim_tA) = 0;
end
