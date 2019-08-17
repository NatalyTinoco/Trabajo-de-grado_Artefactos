# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:07:23 2019

"""
# Leer bbox 
def read_boxes(txtfile):
    import numpy as np
    lines = []
    
    with open(txtfile, "r") as f:
    
        for line in f:
            line = line.strip()
            box = np.hstack(line.split()).astype(np.float)
            box[0] = int(box[0])
            lines.append(box)
    return np.array(lines)