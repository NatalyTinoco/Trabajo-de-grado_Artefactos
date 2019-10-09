# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:29:45 2019

@author: Usuario
"""

import cv2
import skimage.feature
from skimage.feature import greycomatrix, greycoprops
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.filters.rank import median
from skimage.morphology import disk

#entrop=[]
#SSIMN=[]

def GLCM (imA):
    a=int(np.max(imA))
    g = skimage.feature.greycomatrix(imA, [1], [0], levels=a+1, symmetric=False, normed=True)                  
    contraste=skimage.feature.greycoprops(g, 'contrast')[0][0]
    energia=skimage.feature.greycoprops(g, 'energy')[0][0]
    homogeneidad=skimage.feature.greycoprops(g, 'homogeneity')[0][0]
    correlacion=skimage.feature.greycoprops(g, 'correlation')[0][0]
    disimi= greycoprops(g, 'dissimilarity') 
    ASM= greycoprops(g, 'ASM')
    entropia=skimage.measure.shannon_entropy(g) 
    return g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia


def carcDM(imFile):
    HSV=cv2.cvtColor(imFile,cv2.COLOR_RGB2HSV)
    H,cropped,V=cv2.split(HSV)
    cropSinFou=cropped.copy()
    g3,contraste3,energia3,homogeneidad3, correlacion3, disimi3, ASM3,entropia3=GLCM(cropSinFou)
    aa4=cropped.copy()
    bb4=median(aa4, disk(20))
    SSIMN=ssim(bb4,aa4)
    
    return(entropia3,SSIMN)
    
    
