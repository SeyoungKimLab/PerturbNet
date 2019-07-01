function [Lambda_z, Theta_yz, Lambda_y, Theta_xy, stats] = em_scggm(...
    Z, Y, X, ...
    lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy, ...
    options)
% Inputs:
% Z (n samples x r dimensions)
% Y (n_o samples x q dimensions), where n_o <= n
% X (n samples x p dimensions)
% lambdaLambda (regularization for Lambda)
% lambdaTheta (regularization for Theta)
% [options] struct with the following options:
%   - verbose(0): show information or not (0 or 1)
%   - max_em_iters(10): max number of outer iterations
%   - max_M_iters(1): max number of iterations within M-step
%   - sparse_E_step(1) : avoid dense inverse in E step (0 or 1)\n"
%   - sparse_Theta_cd(1) : avoid dense products in ThetaCD (0 or 1)\n"
%   - sigma(1e-4): backtracking termination criterion
%   - em_tol(1e-2): tolerance for terminating EM loop
%   - tol(1e-2): tolerance for terminating M-step
%   - num_blocks_Lambda(-1): number of blocks for Lambda CD
%   - num_blocks_Theta(-1): number of blocks for Theta CD
%   - memory_usage(32000): memory available to process in Mb
%   - num_threads(16): maximum number of threads allowed

    olddir = pwd;
    thisfunc = which(mfilename());
    thisdir = thisfunc(1:end-length('em_scggm.m'));
    cd(thisdir);
    addpath('../MATLAB');
  
    verbose = 0;
    max_em_iters = 10;
    max_M_iters = 2;
    sparse_E_step = 1;
    sparse_Theta_cd = 1;
    sigma = 1.0e-4;
    em_tol = 1.0e-2;
    tol = 1.0e-2;
    num_blocks_Lambda = -1; % >0: manual user input, rather than memory_usage
    num_blocks_Theta = -1; % >0: manual user input, rather than memory_usage
    memory_usage = 32000; % default is 32 Gb
    num_threads = 16;
    if exist('options', 'var')
        if isfield(options, 'verbose')
            verbose = options.verbose;
        end
        if isfield(options, 'max_em_iters')
            max_em_iters = options.max_em_iters;
        end
        if isfield(options, 'max_M_iters')
            max_M_iters = options.max_M_iters;
        end
        if isfield(options, 'sparse_E_step')
            sparse_E_step = options.sparse_E_step;
        end
        if isfield(options, 'sparse_Theta_cd')
            sparse_Theta_cd = options.sparse_Theta_cd;
        end
        if isfield(options, 'sigma')
            refit = options.sigma;
        end
        if isfield(options, 'em_tol')
            em_tol = options.em_tol;
        end
        if isfield(options, 'tol')
            tol = options.tol;
        end
        if isfield(options, 'num_blocks_Lambda')
            num_blocks_Lambda = options.num_blocks_Lambda;
        end
        if isfield(options, 'num_blocks_Theta')
            num_blocks_Theta = options.num_blocks_Theta;
        end
        if isfield(options, 'memory_usage')
            memory_usage = options.memory_usage;
        end
        if isfield(options, 'num_threads')
            num_threads = options.num_threads;
        end
    end
    [n_z, r] = size(Z);
    [n_y, q] = size(Y);
    [n_x, p] = size(X);
    assert(n_x == n_z);

    % Center data - CLI already does this
    %Z = bsxfun(@minus, Z, mean(Z));
    %Y = bsxfun(@minus, Y, mean(Y));
    %X = bsxfun(@minus, X, mean(X));

    dummy = randi(1e6);
    Zfile = sprintf('Z-dummy-%i.txt', dummy);
    Yfile = sprintf('Y-dummy-%i.txt', dummy);
    Xfile = sprintf('X-dummy-%i.txt', dummy);
    dlmwrite(Zfile, Z, 'delimiter', ' ', 'precision', 10);
    dlmwrite(Yfile, Y, 'delimiter', ' ', 'precision', 10);
    dlmwrite(Xfile, X, 'delimiter', ' ', 'precision', 10);

    Lambda_z_file = sprintf('Lambda-z-dummy-%i.txt', dummy);
    Theta_yz_file = sprintf('Theta-yz-dummy-%i.txt', dummy);
    Lambda_y_file = sprintf('Lambda-y-dummy-%i.txt', dummy);
    Theta_xy_file = sprintf('Theta-xy-dummy-%i.txt', dummy);
    stats_file = sprintf('stats-dummy-%i.txt', dummy);
   
    reg_str = sprintf('-Z %g -z %g -Y %g -y %g ', ...
        lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy);
    opts1_str = sprintf('-v %i -I %i -i %i -E %i -T %i ', ...
        verbose, max_em_iters, max_M_iters, sparse_E_step, sparse_Theta_cd);
    opts2_str = sprintf('-s %g -Q %g -q %g -l %i -t %i -m %i -n %i ', ...
        sigma, em_tol, tol, num_blocks_Lambda, num_blocks_Theta, ...
        memory_usage, num_threads);
    in_str = sprintf('%i %i %i %i %i %s %s %s ', ...
        r, q, p, n_z, n_y, Zfile, Yfile, Xfile);
    out_str = sprintf('%s %s %s %s %s ', ...
        Lambda_z_file, Theta_yz_file, Lambda_y_file, Theta_xy_file, stats_file);
    command_str = sprintf('./em_scggm %s %s %s %s %s', ...
        reg_str, opts1_str, opts2_str, in_str, out_str);

    system(command_str);
    try
        Lambda_z = txt_to_sparse(Lambda_z_file);
        Theta_yz = txt_to_sparse(Theta_yz_file);
        Lambda_y = txt_to_sparse(Lambda_y_file);
        Theta_xy = txt_to_sparse(Theta_xy_file);
        stats = txt_to_struct(stats_file);
    catch
        fprintf('%s\n', command_str);
        cd(olddir);
    end
    system(['rm ' Zfile ' ' Yfile ' ' Xfile]);
    system(sprintf('rm %s %s %s %s %s', Lambda_z_file, Theta_yz_file, ...
        Lambda_y_file, Theta_xy_file, stats_file));
    cd(olddir);
end
