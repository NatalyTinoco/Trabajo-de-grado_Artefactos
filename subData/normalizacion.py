# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:26:44 2019

@author: Nataly
"""

def log(img):
    import numpy as np
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img
    