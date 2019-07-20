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
def filtromediana(img):
    B,G,R=cv2.split(img)
    fR = median(R, disk(3))
    fG = median(G, disk(3))
    fB = median(B, disk(3))
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh
def filtrominimo(img):
    B,G,R=cv2.split(img)
    fR = minimum(R, disk(3))
    fG = minimum(G, disk(3))
    fB = minimum(B, disk(3))
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh
def filtrogaussiano(img):
    B,G,R=cv2.split(img)
    fB=cv2.GaussianBlur(B,(3,3),0)
    fR=cv2.GaussianBlur(R,(3,3),0)
    fG=cv2.GaussianBlur(G,(3,3),0)
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh
def filtropromedio(img):
    B,G,R=cv2.split(img)
    fR = mean_bilateral(R.astype(np.uint16), disk(5), s0=10, s1=10)
    fG = mean_bilateral(G.astype(np.uint16), disk(5), s0=10, s1=10)
    fB = mean_bilateral(B.astype(np.uint16), disk(5), s0=10, s1=10)
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh


    

    




