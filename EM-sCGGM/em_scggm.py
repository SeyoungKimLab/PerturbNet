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

def em_scggm(
        Z, Y, X,
        lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy,
        verbose=False, max_em_iters=50, sigma=1e-4, em_tol=1e-2):
    """
    Learns a three-layer network from Z (traits), Y (gene expressions), 
    and X (SNP data). Z and X have n samples, while Y has n_o samples, 
    with n_o <= n. The n_o samples in Y must correspond to 
    the first n_o rows in Z and X.

    Inputs:
      Z: array of shape (n, r)
        The phenotype data matrix with n samples and r traits.
      Y: array of shape (n_o, q)
        The data matrix with n_o samples and q genes.
      X: array of shape (n, p)
        The genotype data matrix with n samples and r traits.
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

    Returns:
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
    """

    olddir = os.getcwd()
    thisdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(thisdir) # move to demo directory
    os.chdir("../EM-sCGGM")

    dummy = rnd.randint(low=0, high=1e6)

    (n, r) = Z.shape
    (n_o, q) = Y.shape
    (n, p) = X.shape

    Zfile = "Z-dummy-%i.txt" % (dummy)
    Yfile = "Y-dummy-%i.txt" % (dummy)
    Xfile = "X-dummy-%i.txt" % (dummy)
    Lambdazfile = "Lambda-z-dummy-%i.txt" % (dummy)
    Thetayzfile = "Theta-yz-dummy-%i.txt" % (dummy)
    Lambdayfile = "Lambda-y-dummy-%i.txt" % (dummy)
    Thetaxyfile = "Theta-xy-dummy-%i.txt" % (dummy)
    statsfile = "stats-dummy-%i.txt" % (dummy)
    np.savetxt(Zfile, Z, fmt="%.10f", delimiter=" ")
    np.savetxt(Yfile, Y, fmt="%.10f", delimiter=" ")
    np.savetxt(Xfile, X, fmt="%.10f", delimiter=" ")
    option_str = "-Z %f -z %f -Y %f -y %f -v %i -i %i -s %f -q %f " % (
        lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy,
        verbose, max_em_iters, sigma, em_tol)
    command_str = "./em_scggm %s   %i %i %i %i %i %s %s %s   %s %s %s %s %s" % (
        option_str,
        r, q, p, n, n_o, Zfile, Yfile, Xfile,
        Lambdazfile, Thetayzfile, Lambdayfile, Thetaxyfile, statsfile)

    ret = os.system(command_str)
    Lambda_z = txt_to_sparse(Lambdazfile)
    Theta_yz = txt_to_sparse(Thetayzfile)
    Lambda_y = txt_to_sparse(Lambdayfile)
    Theta_xy = txt_to_sparse(Thetaxyfile)
    stats = txt_to_dict(statsfile)
    rmline = "rm %s %s %s   %s %s %s %s %s" % (
        Zfile, Yfile, Xfile, Lambdazfile, Thetayzfile, 
        Lambdayfile, Thetaxyfile, statsfile)
    ret = os.system(rmline)
    os.chdir(olddir)
    return (Lambda_z, Theta_yz, Lambda_y, Theta_xy, stats)
