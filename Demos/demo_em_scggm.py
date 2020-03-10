#!/bin/python

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import sksparse.cholmod as skc
import sys

sys.path.append("../EM-sCGGM/")
from em_scggm import em_scggm

n = 100
n_o = 50
r = 50
q = 150
p = 200
Lambda_z = ssp.eye(r, format="coo")
Lambda_z.setdiag(0.3, 1)
Lambda_z.setdiag(0.3, -1)
Lambda_y = ssp.eye(q, format="coo")
Lambda_y.setdiag(0.3, 1)
Lambda_y.setdiag(0.3, -1)

'''
q_influence = int(math.floor(q * 0.5))
Theta_yz_top = ssp.random(q_influence, r)
Theta_yz_bottom = ssp.zeros((q-q_influence,r))
Theta_yz = ssp.vstack([Theta_yz_top, Theta_yz_bottom])
p_influence = int(math.floor(p * 0.5))
Theta_xy_top = ssp.random(p_influence, q)
Theta_xy_bottom = ssp.zeros((p-p_influence,q))
Theta_xy = ssp.vstack([Theta_xy_top, Theta_xy_bottom])
'''
Theta_yz = ssp.random(q, r)
Theta_xy = ssp.random(p, q)

X = np.random.randn(n, p)
meanY = -1*X*Theta_xy*ssl.inv(Lambda_y)
Lambda_y_factor = skc.cholesky(Lambda_y)
noiseY = (Lambda_y_factor.solve_Lt(np.random.randn(q,n))).transpose()
Y = meanY + noiseY
Y_o = Y[0:n_o,:]
meanZ = -1*Y*Theta_yz*ssl.inv(Lambda_z)
Lambda_z_factor = skc.cholesky(Lambda_z)
noiseZ = (Lambda_z_factor.solve_Lt(np.random.randn(r,n))).transpose()
Z = meanZ + noiseZ

lambdaLambda_z = 0.5
lambdaTheta_yz = 1.0
lambdaLambda_y = 0.5
lambdaTheta_xy = 1.0

(estLambda_z, estTheta_yz, estLambda_y, estTheta_xy, estStats) = em_scggm(
    Z, Y, X, lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy)

plt.figure()
plt.spy(Theta_xy)
plt.title("Theta")

plt.figure()
plt.spy(Lambda_y)
plt.title("Lambda")

plt.figure()
plt.spy(estLambda_y)
plt.title("estLambda")

plt.figure()
plt.spy(estTheta_xy)
plt.title("estTheta")

plt.show()
