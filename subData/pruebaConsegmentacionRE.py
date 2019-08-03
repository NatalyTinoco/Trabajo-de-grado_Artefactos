# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:49:08 2019

@author: Nataly
"""

import cv2
import numpy as np

import glob
import pylab as plt 
import pylab as plt 
from matplotlib import pyplot as plt
from rOI import ROI
from normalizacion import log
from filMin import filtrominimo

i=0
melocoton=0
z=0
for imgfile in glob.glob("*.jpg"):
    
    ima='./segROI/#5/Z3/'+imgfile
    imaROI=cv2.imread(ima,0)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    print(imgfile)
    
    im='./segmentacionRE2/nuevaRE/'+imgfile
    imaRE=cv2.imread(im)
    
    imE='./segmentacionRE2/nuevaDM/'+imgfile
    imaDM=cv2.imread(imE)
    
    if str (type(imaRE))=="<class 'NoneType'>":
         imA=imaDM
         
    else:
         imA=imaRE

    
       
   
    YUV=cv2.cvtColor(imA,cv2.COLOR_RGB2HSV)
    
    #YUV=cv2.cvtColor(imA,cv2.COLOR_RGB2YUV)
    """
    for z in range(3):
        YUV[:,:,z]=YUV[:,:,z]*imaROI
    """
    Y,U,V=cv2.split(YUV)    
   
    #Y = cv2.normalize(Y, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #U= cv2.normalize(U, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #V= cv2.normalize(V, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    #imaROI=imaROI*-1    
    #Y=Y*imaROI
    #U=U*imaROI
    #V=V*imaROI
    """
    cv2.imshow('ima',imA)
    cv2.waitKey(0)
    
    cv2.imshow('Y',Y)
    cv2.waitKey(0)
    cv2.imshow('U',U)
    cv2.waitKey(0)
    cv2.imshow('V',V)
    cv2.waitKey(0)
    #"""
    cv2.destroyAllWindows()
    umbral1=0.35*255
    umbral2=0.75*255
    ta=V.shape
    ta=list(ta)
    binary=V.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if U[f,c]<umbral1 and V[f,c]>umbral2:
                binary[f,c]=255
            else:
                binary[f,c]=0
    binary=binary*imaROI
    """
    cv2.imshow('image',binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #"""
    
    #binary=binary*-1
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('image',openi)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #15-15
    #25-25
    #openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
   
    openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dilatacion = cv2.dilate(openi,kernel,iterations = 1)
    """
    cv2.imshow('image',dilatacion)
    cv2.waitKey(0)
    #"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    close=cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
   
    #openi = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
   
    """
    cv2.imshow('image',close)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
          
    if str (type(imaRE))=="<class 'NoneType'>":
        dire='./segmentacionCombinacionRE2yArticulo/DM/'+imgfile
        cv2.imwrite(dire,close)
        print(str (type(imaRE)))  
    else:
        direR='./segmentacionCombinacionRE2yArticulo/RE/'+imgfile
        cv2.imwrite(direR,close)
    
    