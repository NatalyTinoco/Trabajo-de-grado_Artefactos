# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:06:05 2019

@author: Nataly
"""

import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')

import cv2
import numpy as np 
from rOI import ROI
def adaptativeequalization(img):  
    imR, imG, imB=cv2.split(img) 
    #Adaptative Equalization
    clahe=cv2.createCLAHE(2,(8,8))
    imhaR=clahe.apply(imR)
    imhaG=clahe.apply(imG)
    imhaB=clahe.apply(imB)

    imha=cv2.merge((imhaR,imhaG,imhaB))

    return imha
import matplotlib.pyplot as plt
img=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subDM/subNormDM/00063_batch2.jpg')  
imaROI=ROI(img)
img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
#img=adaptativeequalization(img)    
#qqq1=cv2.resize(img,(600,500))
#cv2.imshow('draw',qqq1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
img= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#plt.imshow(mostrar)
#plt.show()
R,G,B=cv2.split(mostrar)
qqq=cv2.resize(G,(600,500))
cv2.imshow('draw',qqq)
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.imshow(G,'Greys')
#plt.show()
imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
imaROIm = cv2.normalize(imaROI, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
qqq=cv2.resize(imaROIm,(600,500))
cv2.imshow('draw',qqq)
cv2.waitKey(0)
cv2.destroyAllWindows()  
for z in range(3):
    img[:,:,z]=img[:,:,z]*imaROI

R,G,B=cv2.split(img)
qqq=cv2.resize(G*imaROI,(600,500))
cv2.imshow('draw',qqq)
cv2.waitKey(0)
cv2.destroyAllWindows()  



    
    
    
    