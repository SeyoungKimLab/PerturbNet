#!/bin/python

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import sys

sys.path.append("../Mega-sCGGM/")
from mega_scggm import mega_scggm

n = 100
p = 200
q = 150
Lambda = ssp.eye(q, format="coo")
Lambda.setdiag(0.3, 1)
Lambda.setdiag(0.3, -1)
p_influence = int(math.floor(p * 0.5))
Theta_top = ssp.random(p_influence, q)
Theta_bottom = ssp.coo_matrix((p-p_influence,q))
Theta = ssp.vstack([Theta_top, Theta_bottom])
X = np.random.randn(n, p)
Sigma = ssl.inv(Lambda)
meanY = -1 * X @ Theta @ Sigma
try:
    import sksparse.cholmod as skc
    Lambda_factor = skc.cholesky(Lambda)
    noiseY = (Lambda_factor.solve_Lt(np.random.randn(q,n))).transpose()
except:
    noiseY = np.random.multivariate_normal(np.zeros(q), Sigma.todense(), size=n)
Y = meanY + noiseY

lambdaLambda = 0.5
lambdaTheta = 1.0
(estLambda, estTheta, estStats) = mega_scggm(Y, X, lambdaLambda, lambdaTheta)

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
