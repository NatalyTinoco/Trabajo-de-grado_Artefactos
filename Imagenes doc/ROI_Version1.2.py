# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:37:26 2019

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
import glob

for imgfile in glob.glob("*.jpg"):
    ima=read_img(imgfile)
    plt.imshow(ima)
    plt.show()
    imB, imG, imR=cv2.split(ima)
    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2HSV)
    I,I,II=cv2.split(imA)
    ta=II.shape
    ta=list(ta)
    binary=II.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if II[f,c]<=0.003:
                binary[f,c]=0
            else:
                binary[f,c]=1
                    
    binary = (binary*255).astype(np.uint8)
    plt.imshow(binary, cmap=plt.cm.gray)
    plt.show()
    ### Transformaciones Morfologicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #dilation = cv2.dilate(opening,kernel,iterations = 1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #close=cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    plt.imshow(close, cmap=plt.cm.gray)
    plt.show()
   

    """
    _,contours,_ = cv2.findContours(close,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(ima,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Show",ima)
    #plt.imshow(im)
    #plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #"""

        """
    fig, ax = try_all_threshold(II, figsize=(10, 8), verbose=False)
    plt.show()
    
      
    #thresh = threshold_li(II)
    #thresh=threshold_minimum(II)
    thresh=threshold_otsu(II)
    #thresh=threshold_triangle(II)
    #ret3,thresh= cv2.threshold(II,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = II > thresh
    """


