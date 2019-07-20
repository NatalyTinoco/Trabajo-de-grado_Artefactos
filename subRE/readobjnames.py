# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:18:22 2019
"""

def read_obj_names(textfile):
    
    import numpy as np 
    classnames = []
    
    with open(textfile) as f:
        for line in f:
            line = line.strip('\n')
            if len(line)>0:
                classnames.append(line)
            
    return np.hstack(classnames)