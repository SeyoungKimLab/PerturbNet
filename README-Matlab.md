## EM-sCGGM

The em_scggm(...) function is found in the EM-sCGGM/ directory.

```
function [Lambda_z, Theta_yz, Lambda_y, Theta_xy, stats] = em_scggm(...
    Z, Y, X, ... 
    lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy, ... 
    options)
% Inputs:
% Z: n samples x r features (eg traits)
% Y: n_o samples x q features (eg genes), where n_o <= n
% X: n samples x p features (eg SNPs)
% lambdaLambda_z: regularization for Lambda_z
% lambdaTheta_yz: regularization for Theta_yz
% lambdaLambda_y: regularization for Lambda_y
% lambdaTheta_xy: regularization for Theta_xy 
% options: (optional) struct with the following fields:
%   - verbose(0): show information or not (0 or 1)
%   - max_em_iters(10): max number of outer iterations
%   - max_M_iters(1): max number of iterations within M-step
%   - sparse_E_step(1) : avoid dense inverse in E step (0 or 1)
%   - sparse_Theta_cd(1) : avoid dense products in ThetaCD (0 or 1)
%   - sigma(1e-4): backtracking termination criterion
%   - em_tol(1e-2): tolerance for terminating EM loop
%   - tol(1e-2): tolerance for terminating M-step
%   - num_blocks_Lambda(-1): number of blocks for Lambda CD
%   - num_blocks_Theta(-1): number of blocks for Theta CD
%   - memory_usage(32000): memory available to process in Mb
%   - num_threads(16): maximum number of threads allowed
%
% Outputs:
% Lambda_z: r x r sparse matrix (eg trait network)
% Theta_yz: q x r sparse matrix (eg gene-trait mapping)
% Lambda_y: q x q sparse matrix (eg gene network)
% Theta_xy: p x q sparse matrix (eg SNP-gene mapping)
% stats: struct with at least the following fields:
%   - objval: history of EM objective over iterations
%   - time: total walltime in seconds from the start
```

## Fast-sCGGM

The fast_scggm(...) function is found in the Fast-sCGGM/ directory.

```
function [Lambda, Theta, stats] = fast_scggm(...
    Y, X, lambdaLambda, lambdaTheta, options)
% Inputs:
% Y: n samples x q features (eg genes)
% X: n samples x p features (eg SNPs)
% lambdaLambda: regularization for Lambda
% lambdaTheta: regularization for Theta
% options: (optional) struct with the following fields:
%   - verbose(0): show information or not (0 or 1)
%   - max_iters(10): max number of outer iterations
%   - sigma(1e-4): backtracking termination criterion
%   - tol(1e-2): tolerance for terminating outer loop
%   - Lambda0(none): q x q sparse matrix to initialize Lambda
%   - Theta0(none): p x q sparse matrix to initialize Theta
%   - refit(0): refit selected model without adding any edges
%
% Outputs:
% Lambda: q x q sparse matrix (eg gene network)
% Theta: p x q sparse matrix (eg SNP-gene mapping)
% stats: struct with at least the following fields:
%   - objval: history of EM objective over iterations
%   - time: total walltime in seconds from the start
```

## Mega-sCGGM

The mega_scggm(...) function is found in the Mega-sCGGM/ directory.

```
function [Lambda, Theta, stats] = mega_scggm(...                                    
    Y, X, lambdaLambda, lambdaTheta, options)                                       
% Inputs:                                                                           
% Y: n samples x q features (eg genes)                                              
% X: n samples x p features (eg SNPs)                                               
% lambdaLambda: regularization for Lambda                                           
% lambdaTheta: regularization for Theta                                             
% options: (optional) struct with the following fields:                             
%   - Lambda0(none): q x q sparse matrix to initialize Lambda                       
%   - Theta0(none): p x q sparse matrix to initialize Theta                         
%   - verbose(0): show information or not (0 or 1)                                  
%   - max_iters(10): max number of outer iterations                                 
%   - sigma(1e-4): backtracking termination criterion                               
%   - tol(1e-2): subgradient/L1norm tolerance for terminating outer loop            
%   - obj_tol(1.0e-13): CG tolerance for calculating objective function             
%   - grad_tol(1.0e-10): CG tolerance for calculating gradient                      
%   - hess_tol(1.0e-8): CG tolerance for calculating hessian                        
%   - num_blocks_Lambda(-1): number of blocks for Lambda CD                         
%   - num_blocks_Theta(-1): number of blocks for Theta CD                           
%   - memory_usage(32000): memory available to process in Mb                        
%   - max_threads(4): max number of threads to use                                  
%   - refit(0): update (Lambda0,Theta0) without adding any edges                    
%                                                                                   
% Outputs:                                                                          
% Lambda: q x q sparse matrix (eg gene network)                                     
% Theta: p x q sparse matrix (eg SNP-gene mapping)                                  
% stats: struct with at least the following fields:                                 
%   - objval: history of EM objective over iterations                               
%   - time: total walltime in seconds from the start
```
