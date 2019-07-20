# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 18:02:29 2019

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
def log(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img


for imgfile in glob.glob("*.jpg"):
    ima=read_img(imgfile)
    
    """ # Primera forma #"""
    #imR, imG, II=cv2.split(ima)
    """ #Segunda forma #"""
    #imA=cv2.cvtColor(ima,cv2.COLOR_RGB2HSV)
    #I,I,II=cv2.split(imA)
    """ # Tercera forma #"""   
    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2XYZ)
    I,I,II=cv2.split(imA)
    
    #II=rank.entropy(II, disk(10))
    #II=II.astype(np.uint8)
    #umbral=np.min(entropia)
    #clahe=cv2.createCLAHE(2,(15,15))
    #II=clahe.apply(II)
    
    ta=II.shape
    ta=list(ta)
    #plt.imshow(II, cmap=plt.cm.gray)
    #plt.show()        
    MIN=int(np.min(II))
    MAX=int(np.max(II))
  
    hist = cv2.calcHist([II],[0],None,[256],[0,255])
    #plt.plot(hist)
    #plt.show()
    
    div=6
    nuevohist=hist.tolist() 
    l=int(len(nuevohist)/div)
    i=1
    valm=0
    hasta=0
    for y in range(1,div+1):
        suma=sum(nuevohist[i:int(l)*y])
        if suma>valm:
            valm=suma
            hasta=y
        #print(i,l)
        #print(suma,y)
        i=int(l)*y+1
       
    print(hasta*l)
    porcen=0.28
    a=hist[:int((l*hasta)*porcen)]
    a=a.tolist()         
    hist=hist.tolist() 
    u=np.max(a)
    umbral1 = hist.index(u)
    a=a[umbral1:]
    
    uu=np.min(a)
    histb=hist[umbral1:]
    
    umbral=histb.index(uu)
    #print(umbral)
    umbral=umbral+len(hist[:umbral1])
    binary=II.copy()
    print(umbral)
    for f in range(ta[0]):
        for c in range (ta[1]):
            if II[f,c]<=umbral:
                binary[f,c]=0
            else:
                binary[f,c]=1

    binary = (binary*255).astype(np.uint8)
    plt.imshow(binary, cmap=plt.cm.gray)
    plt.show()
    #""" 
    ### Transformaciones Morfologicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    close=cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
    openi = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    #dilation = cv2.dilate(opening,kernel,iterations = 1)

    #plt.imshow(close, cmap=plt.cm.gray)
    #plt.show()    
    #plt.imshow(openi, cmap=plt.cm.gray)
    #plt.show()
    
    #"""

    dire='./segROI/#6/V/'+imgfile
    #img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)   
    cv2.imwrite(dire,openi)
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
