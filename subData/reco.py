# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:24:34 2019

@author: Nataly
"""
import cv2
import numpy as np

import glob
import pylab as plt 
import pylab as plt 
from matplotlib import pyplot as plt
i=0
melocoton=0
z=0
for imgfile in glob.glob("*.jpg"):
    ima='./segROI/#5/Z3/'+imgfile
    imaROI=cv2.imread(ima,0)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    print(imgfile)
    
    im='./segmentacionRE2/RE/'+imgfile
    imaRE=cv2.imread(im,0)
    imaRE = cv2.normalize(imaRE, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    imE='./segmentacionRE2/DM/'+imgfile
    imaDM=cv2.imread(imE,0)
    imaDM = cv2.normalize(imaDM, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    imagen=cv2.imread(imgfile)
    for z in range(3):
        imagen[:,:,z]=imagen[:,:,z]*imaROI
   
    """cv2.imshow('image',imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    for z in range(3):
        if str (type(imaRE))=="<class 'NoneType'>":
            imagen[:,:,z]=imagen[:,:,z]*imaDM
        else:
            imagen[:,:,z]=imagen[:,:,z]*imaRE
            
        
    if str (type(imaRE))=="<class 'NoneType'>":
        dire='./segmentacionRE2/nuevaDM/'+imgfile
        cv2.imwrite(dire,imagen)
        print(str (type(imaRE)))  
    else:
        direR='./segmentacionRE2/nuevaRE/'+imgfile
        cv2.imwrite(direR,imagen)
    #cv2.imshow('image',imagen)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()