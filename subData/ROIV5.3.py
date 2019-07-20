# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:55:15 2019

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
    
    
    ta=II.shape
    ta=list(ta)
    #plt.imshow(II, cmap=plt.cm.gray)
    #plt.show()        
    MIN=int(np.min(II))
    MAX=int(np.max(II))
  
    hist = cv2.calcHist([II],[0],None,[MAX+1],[MIN,MAX])
    #hist = cv2.calcHist([II],[0],None,[256],[0,255])
    #plt.plot(hist)
    #plt.show()
    
    div=8
    #div=6
    nuevohist=hist.tolist() 
    l=int(len(nuevohist)/div)
    i=1
    a=0
    suma=np.zeros((div))
    valm=0
    hasta=0
    menor=0
    for y in range(1,div+1):
        suma[a]=sum(nuevohist[i:int(l)*y])
        if suma[a]>valm:
            valm=suma[a]
            hasta=y
        i=int(l)*y+1
        a=a+1
    suma=suma[:hasta].tolist()
    h=np.min(suma)
    menor=suma.index(h)
    #print(hasta)
    #print(menor)
    porcen=(menor+1)/div
    #print(porcen)
    umbral=(l*(menor+1))*porcen
    binary=II.copy()
    #print(umbral)
    for f in range(ta[0]):
        for c in range (ta[1]):
            if II[f,c]<=umbral:
                binary[f,c]=0
            else:
                binary[f,c]=1

    binary = (binary*255).astype(np.uint8)
    #plt.imshow(binary, cmap=plt.cm.gray)
    #plt.show()
    #""" 
    ### Transformaciones Morfologicas
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22, 22))
    close=cv2.morphologyEx(openi, cv2.MORPH_CLOSE, kernel)

    #plt.imshow(close, cmap=plt.cm.gray)
    #plt.show()    
    #plt.imshow(openi, cmap=plt.cm.gray)
    #plt.show()
    
    #"""

    dire='./segROI/#5/Z/'+imgfile
    #img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)   
    cv2.imwrite(dire,close)
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
    print(imgfile)
