# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:46:43 2019

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
from skimage.filters import threshold_otsu, threshold_local

from PIL import ImageEnhance
from PIL import Image 
import glob
from skimage.segmentation import flood, flood_fill

for imgfile in glob.glob("*.jpg"):
    ima=read_img(imgfile)
    imR, imG, imB=cv2.split(ima)
    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2HSV)
    I,I,II=cv2.split(imA)
    
    #fig, ax = try_all_threshold(II, figsize=(10, 8), verbose=False)
    #plt.show()  
    #block_size = 51
    #thresh = threshold_local(II, block_size, offset=10)
    #thresh = threshold_li(II)
    thresh=threshold_minimum(II)
    #thresh=threshold_otsu(II)
    #thresh=threshold_triangle(II)
    #ret3,thresh= cv2.threshold(II,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = II > thresh
    #"""
                    
    binary = (binary*255).astype(np.uint8)
    
    plt.imshow(binary, cmap=plt.cm.gray)
    plt.show()
    
    

    #"""
    ### Transformaciones Morfologicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #dilation = cv2.dilate(opening,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    close=cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    plt.imshow(close, cmap=plt.cm.gray)
    plt.show()
    
    dire='./segROI/#3/'+imgfile
    #img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)   
    #cv2.imwrite(dire,close)
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
    #"""
     
  

    
    """
    im2, contornos, hierarchy = cv2.findContours (binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours (ima, contornos, -1, (0,255,0), 3)
    plt.imshow(ima, cmap=plt.cm.gray)
    plt.show()
    
   
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
