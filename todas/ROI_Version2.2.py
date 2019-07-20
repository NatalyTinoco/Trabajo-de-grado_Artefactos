# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 20:04:57 2019

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

def log(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img

for imgfile in glob.glob("*.jpg"):
    ima=read_img(imgfile)
    ima=log(ima)
    imR, imG, imB=cv2.split(ima)
    
    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2HSV)
    I,I,II=cv2.split(imA)
    #II=imB
    ta=II.shape
    ta=list(ta)
    umbral0=np.mean(II)
    umbral1=np.max(II)
    umbral2=np.min(II)
    umbral=umbral1*0.025
    #plt.imshow(II, cmap=plt.cm.gray)
    #plt.show()
    binary=II.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if II[f,c]<=umbral:
                binary[f,c]=0
            else:
                binary[f,c]=1
                    
    binary = (binary*255).astype(np.uint8)
    #plt.imshow(binary, cmap=plt.cm.gray)
    #plt.show()
    ### Transformaciones Morfologicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (90, 90))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #dilation = cv2.dilate(opening,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close=cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    #plt.imshow(close, cmap=plt.cm.gray)
    #plt.show()
   
    dire='./segROI/#7/'+imgfile
    #img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)   
    cv2.imwrite(dire,close)
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
    
 