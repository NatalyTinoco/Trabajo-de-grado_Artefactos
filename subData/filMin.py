# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:27:26 2019

@author: Nataly
"""

def filtrominimo(img):
    from skimage.filters.rank import minimum
    from skimage.morphology import disk
    import cv2
    B,G,R=cv2.split(img)
    fR = minimum(R, disk(3))
    fG = minimum(G, disk(3))
    fB = minimum(B, disk(3))
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh
