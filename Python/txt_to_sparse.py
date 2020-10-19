#!/bin/python

import csv
import scipy.sparse as ssp

def txt_to_sparse(filename):
    with open(filename, "r") as tf: 
        tfr = csv.reader(tf, delimiter=" ")
        first_row = next(tfr)
        p = int(first_row[0])
        q = int(first_row[1])
        nnz = int(first_row[2])
        the_rest = [(int(row[0])-1, int(row[1])-1, float(row[2]))
                    for row in tfr]
    (i, j, data) = zip(*the_rest)
    Theta = ssp.coo_matrix((list(data), (list(i),list(j))), shape=(p, q))
    return Theta
