#!/bin/python

import os
import numpy as np
import numpy.random as rnd
import scipy.sparse as ssp
import sys

sys.path.append("../Python/")
from txt_to_sparse import txt_to_sparse
from txt_to_dict import txt_to_dict
from sparse_to_txt import sparse_to_txt

def mega_scggm(
        Y, X, lambdaLambda, lambdaTheta, 
        verbose=False, max_iters=50, sigma=1e-4, tol=1e-2,
        num_blocks_Lambda=-1, num_blocks_Theta=-1, memory_usage=32000,
        threads=16, refit=False, Lambda0=None, Theta0=None):
    """
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
    """

    olddir = os.getcwd()
    thisdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(thisdir) # move to demo directory
    os.chdir("../Mega-sCGGM")

    dummy = rnd.randint(low=0, high=1e6)
    L0_str = ""
    T0_str = ""
    if Lambda0:
        Lambda0file = "Lambda0-dummy-%i.txt" % (dummy)
        sparse_to_txt(Lambda0file, Lambda0)
        L0_str = "-L \"%s\" " % (Lambda0file)
    if Theta0:
        Theta0file = "Theta0-dummy-%i.txt" % (dummy)
        sparse_to_txt(Theta0file, Theta0)
        T0_str = "-T \"%s\" " % (Theta0file)


    (n_y, q) = Y.shape
    (n_x, p) = X.shape

    Yfile = "Y-dummy-%i.txt" % (dummy)
    Xfile = "X-dummy-%i.txt" % (dummy)
    Lambdafile = "Lambda-dummy-%i.txt" % (dummy)
    Thetafile = "Theta-dummy-%i.txt" % (dummy)
    statsfile = "stats-dummy-%i.txt" % (dummy)
    np.savetxt(Yfile, Y, fmt="%.10f", delimiter=" ")
    np.savetxt(Xfile, X, fmt="%.10f", delimiter=" ")
    mega_str = "-l %i -t %i -m %i -n %i " % (
        num_blocks_Lambda, num_blocks_Theta, memory_usage, threads)
    option_str = "-y %f -x %f -v %i -i %i -s %f -q %f -r %i  %s  %s %s " % (
        lambdaLambda, lambdaTheta,
        verbose, max_iters, sigma, tol, refit, 
        mega_str, L0_str, T0_str)
    command_str = "./mega_scggm %s   %i %i %i %i %s %s   %s %s %s" % (
        option_str,
        n_y, q, n_x, p, Yfile, Xfile,
        Lambdafile, Thetafile, statsfile)
    print(command_str)

    ret = os.system(command_str)
    Lambda = txt_to_sparse(Lambdafile)
    Theta = txt_to_sparse(Thetafile)
    stats = txt_to_dict(statsfile)
    rmline = "rm %s %s %s %s %s" % (Yfile, Xfile, Lambdafile, Thetafile, statsfile)
    ret = os.system(rmline)
    os.chdir(olddir)
    return (Lambda, Theta, stats)

