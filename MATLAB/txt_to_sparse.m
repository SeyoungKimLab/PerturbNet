function [A] = txt_to_sparse(filename)
% Reads sparse matrix files in the formatused by Perturb-Net
% Input: file location of matrix
% Output: sparse matrix
    data = dlmread(filename);
    A = sparse(data(2:end,1), data(2:end,2), data(2:end,3), ...
        data(1,1), data(1,2));
end
