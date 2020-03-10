#!/bin/python

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import sksparse.cholmod as skc
import sys

sys.path.append("../Fast-sCGGM/")
from fast_scggm import fast_scggm

n = 100
p = 200
q = 150
Lambda = ssp.eye(q, format="coo")
Lambda.setdiag(0.3, 1)
Lambda.setdiag(0.3, -1)
p_influence = int(math.floor(p * 0.5))
Theta_top = ssp.random(p_influence, q)
Theta_bottom = ssp.zeros((p-p_influence,q))
Theta = ssp.vstack([Theta_top, Theta_bottom])
X = np.random.randn(n, p)
meanY = -1*X*Theta*ssl.inv(Lambda)
Lambda_factor = skc.cholesky(Lambda)
noiseY = (Lambda_factor.solve_Lt(np.random.randn(q,n))).transpose()
Y = meanY + noiseY

lambdaLambda = 0.5
lambdaTheta = 1.0
(estLambda, estTheta, estStats) = fast_scggm(Y, X, lambdaLambda, lambdaTheta)

plt.figure()
plt.spy(Theta)
plt.title("Theta")

plt.figure()
plt.spy(Lambda)
plt.title("Lambda")

plt.figure()
plt.spy(estLambda)
plt.title("estLambda")

plt.figure()
plt.spy(estTheta)
plt.title("estTheta")


plt.show()
