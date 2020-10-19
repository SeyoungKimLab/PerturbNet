## EM-sCGGM

```
Help on module em_scggm:

NAME
    em_scggm

FUNCTIONS
    em_scggm(
             Z, Y, X, 
             lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy, 
             verbose=False, max_em_iters=50, sigma=0.0001, em_tol=0.01)
        Learns a three-layer network from Z (traits), Y (gene expressions), 
        and X (SNP data). Z and X have n samples, while Y has n_o samples, 
        with n_o <= n. The n_o samples in Y must correspond to 
        the first n_o rows in Z and X.
        
        Parameters
        ----------
        
        Z: array of shape (n, r)
            The data matrix with n samples and r traits.
        
        Y: array of shape (n_o, q)
            The data matrix with n_o samples and q genes.
        
        X: array of shape (n, p)
        
        lambdaLambda_z: float > 0
            regularization for Lambda_z
        
        lambdaTheta_yz: float > 0
            regularization for Theta_yz
        
        lambdaLambda_y: float > 0
            regularization for Lambda_y
          
        lambdaTheta_xy: float > 0
            regularization for Theta_xy
        
        verbose: bool (optional, default=False)
            print information or not
        
        max_em_iters: int (optional, default=50)
            max number of EM iterations
        
        sigma: float > 0 (optional, default=1e-4)
            backtracking termination criterion
        
        em_tol: float > 0 (optional, default=1e-2)
            tolerance for terminating outer loop
        
        Returns
        -------
        
        Lambda_z: scipy.sparse array of shape (r, r)
            Trait network parameters
        
        Theta_yz: scipy.sparse array of shape (q, r)
            Gene-trait mapping parameters
        
        Lambda_y: scipy.sparse array of shape (r, r)
            Gene network parameters
        
        Theta_xy: scipy.sparse array of shape (r, r)
            SNP-gene mapping parameters
        
        stats_dict: dict
            dictionary with info about optimization, including:
            objval: objective at each iteration
            time: total walltime after each iteration

FILE
    Perturb-Net/EM-sCGGM/em_scggm.py
```

## Mega-sCGGM

```
Help on module mega_scggm:

NAME
    mega_scggm

FUNCTIONS
    mega_scggm(Y, X, lambdaLambda, lambdaTheta, verbose=False, max_iters=50, sigma=0.0001, tol=0.01, num_blocks_Lambda=-1, num_blocks_Theta=-1, memory_usage=32000, threads=16, refit=False, Lambda0=None, Theta0=None)
        Args:
          Y: output data matrix (n samples x q dimensions target variables)
          X: input data matrix (n samples x p dimensions covariate variables)
          lambdaLambda: regularization for Lambda_y
          lambdaTheta: regularization for Theta_xy
        Optional args:
          verbose: print information or not
          max_iters: max number of outer iterations
          sigma: backtracking termination criterion
          tol: tolerance for terminating outer loop
          num_blocks_Lambda: number of blocks for Lambda CD
          num_blocks_Theta: number of blocks for Theta CD
          memory_usage: memory capacity in MB
          threads: the maximum number of threads
          refit: refit (Lambda0, Theta0) without adding any edges
          Lambda0: q x q scipy.sparse matrix to initialize Lambda
          Theta0: p x q scipy.sparse matrix to initialize Theta
        
        Returns:
            Lambda: q x q sparse matrix
            Theta: p x q sparse matrix
            stats_dict: dict of logging results

FILE
    PerturbNet/Mega-sCGGM/mega_scggm.py
    
```

## Fast-sCGGM

```
Help on module fast_scggm:

NAME
    fast_scggm

FUNCTIONS
    fast_scggm(Y, X, lambdaLambda, lambdaTheta, verbose=False, max_iters=50, sigma=0.0001, tol=0.01, refit=False, Lambda0=None, Theta0=None)
        Inputs:
          Y (n samples x q dimensions target variables)
          X (n samples x p dimensions covariate variables)
          lambdaLambda (regularization for Lambda_y)
          lambdaTheta (regularization for Theta_xy)
        Optional inputs:
          verbose: print information or not
          max_iters: max number of outer iterations
          sigma: backtracking termination criterion
          tol: tolerance for terminating outer loop
          refit: refit (Lambda0, Theta0) without adding any edges
          Lambda0: q x q scipy.sparse matrix to initialize Lambda
          Theta0: p x q scipy.sparse matrix to initialize Theta
        Returns (Lambda, Theta, stats_dict)

FILE
    PerturbNet/Fast-sCGGM/fast_scggm.py
```
