#!/bin/python

def partition_to_ordering(parts, includezero=True):
    num_items = len(parts)
    num_parts = len(set(parts))
    sorted_items = sorted(enumerate(parts), key=lambda x: x[1])
    if includezero:
        sorted_ixs = [x[0] for x in sorted_items]
        sorted_parts = [x[1] for x in sorted_items]
        return sorted_ixs
    else:
        sorted_ixs = [x[0] for x in sorted_items if x[1]>0]
        return sorted_ixs
