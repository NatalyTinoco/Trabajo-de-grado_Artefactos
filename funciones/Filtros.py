# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 21:44:23 2019

@author: Nataly
"""

import cv2
import numpy as np
from skimage.filters.rank import autolevel
from skimage.filters.rank import median
from skimage.filters.rank import minimum
from skimage.filters import rank
from skimage.filters.rank import mean_bilateral

from skimage.morphology import disk

def filtroautolevel(img):
    B,G,R=cv2.split(img)
    fR = autolevel(R.astype(np.uint16), disk(7))
    fG = autolevel(G.astype(np.uint16), disk(7))
    fB = autolevel(B.astype(np.uint16), disk(7))
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh
def filtromediana(img,a):
    B,G,R=cv2.split(img)
    fR = median(R, disk(a))
    fG = median(G, disk(a))
    fB = median(B, disk(a))
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh
def filtrominimo(img,a):
    B,G,R=cv2.split(img)
    fR = minimum(R, disk(a))
    fG = minimum(G, disk(a))
    fB = minimum(B, disk(a))
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh
def filtrogaussiano(img,a):
    B,G,R=cv2.split(img)
    fB=cv2.GaussianBlur(B,(a,a),0)
    fR=cv2.GaussianBlur(R,(a,a),0)
    fG=cv2.GaussianBlur(G,(a,a),0)
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh
def filtropromedio(img,a):
    B,G,R=cv2.split(img)
    fR = mean_bilateral(R.astype(np.uint16), disk(a), s0=10, s1=10)
    fG = mean_bilateral(G.astype(np.uint16), disk(a), s0=10, s1=10)
    fB = mean_bilateral(B.astype(np.uint16), disk(a), s0=10, s1=10)
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh


    

    




