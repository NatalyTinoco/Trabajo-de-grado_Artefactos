# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:35:27 2019

@author: Daniela
"""

# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import cv2
from skimage import exposure
import numpy as np
from skimage.morphology import disk
from skimage.filters import rank

def globalequalization(img):
    imR, imG, imB=cv2.split(img) 
    #Global Equaliation
    imhR=cv2.equalizeHist(imR)
    imhG=cv2.equalizeHist(imG)
    imhB=cv2.equalizeHist(imB)
    
    imh=cv2.merge((imhR,imhG,imhB))
    return imh

def adaptativeequalization(img):  
    imR, imG, imB=cv2.split(img) 
    #Adaptative Equalization
    clahe=cv2.createCLAHE(2,(8,8))
    imhaR=clahe.apply(imR)
    imhaG=clahe.apply(imG)
    imhaB=clahe.apply(imB)

    imha=cv2.merge((imhaR,imhaG,imhaB))
    
    return imha

def contraststretching(img):
    #contrast Stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale