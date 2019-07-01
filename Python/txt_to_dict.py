#!/bin/python

import csv
import numpy as np

def txt_to_dict(f):
    stats_dict = {}
    with open(f, "r") as sf:
        sfr = csv.reader(sf, delimiter=" ")
        for row in sfr:
            stats_dict[row[0]] = np.array([float(x) for x in row[1:] if x])
    return stats_dict

