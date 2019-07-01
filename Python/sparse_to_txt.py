#!/bin/python

import scipy.sparse as ssp

def sparse_to_txt(filename, A):
    (i,j,v) = ssp.find(A)
    (p, q) = A.shape
    nnz = i.shape[0]
    with open(filename, "w") as tf:
        tf.write("%i %i %i\n" % (p,q,nnz))
        for n in range(0,nnz):
            tf.write("%i %i %.13f\n" % (i[n],j[n],v[n]))
