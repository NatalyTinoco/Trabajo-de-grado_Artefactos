# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:25:02 2019

@author: Nataly
"""

import cv2
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np 
from pylab import *
from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from skimage.morphology import disk

from skimage.filters import threshold_otsu
from skimage.filters import threshold_li
from skimage.filters import threshold_minimum
from skimage.filters import threshold_triangle
from skimage.filters import try_all_threshold
from PIL import ImageEnhance
from PIL import Image 
from skimage.filters import rank

import glob

for imgfile in glob.glob("*.jpg"):
    ima=read_img(imgfile)
    """ # Primera forma #"""
    imR, imG, II=cv2.split(ima)
    """ #Segunda forma #"""
    #imA=cv2.cvtColor(ima,cv2.COLOR_RGB2HSV)
    #I,I,II=cv2.split(imA)
    """ # Tercera forma #"""   
    #imA=cv2.cvtColor(ima,cv2.COLOR_RGB2XYZ)
    #I,I,II=cv2.split(imA)
    
    II=rank.entropy(II, disk(15))
    II=II.astype(np.uint8)
    
    ta=II.shape
    ta=list(ta)
    plt.imshow(II, cmap=plt.cm.gray)
    plt.show()   
     
    MIN=int(np.min(II))
    MAX=int(np.max(II))
  
    hist = cv2.calcHist([II],[0],None,[MAX],[MIN,MAX])
    plt.plot(hist)
    plt.show()
    umbral=MAX*0.3
    binary=II.copy()
    print(umbral)
    for f in range(ta[0]):
        for c in range (ta[1]):
            if II[f,c]<=umbral:
                binary[f,c]=0
            else:
                binary[f,c]=1

    binary = (binary*255).astype(np.uint8)
    
    """
    umbral=np.mean(entropia)
    #plt.imshow(entropia, cmap=plt.cm.gray)
    #plt.show()
    umbral=umbral*0.3
    print(umbral)
    binary=II.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if II[f,c]<=umbral:
                binary[f,c]=0
            else:
                binary[f,c]=1
    
         
    binary = (binary*255).astype(np.uint8)
    """
    #plt.imshow(binary, cmap=plt.cm.gray)
    #plt.show()
    ### Transformaciones Morfologicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #dilation = cv2.dilate(opening,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    close=cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    plt.imshow(close, cmap=plt.cm.gray)
    plt.show()
   
    dire='./segROI/#4/'+imgfile
    #img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)   
    cv2.imwrite(dire,close)
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()